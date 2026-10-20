"""Steps 1-3: a PV energy community under one local price.

Step 1  rho in the realistic range 0.7-1 (same clouds over the same street),
        plus rho = 0 as a reference.
Step 2  two price rules: "fixed" (the paper's PV-shaped price) vs "sdr"
        (set every step by the community's supply and demand -> feedback).
Step 3  battery power swept against a fixed feeder limit, with PV counted
        in the feeder flow, to see when charging alone breaks the feeder.

Run from the repo root:   python -m experiments.run_community
Outputs: results/community/raw.csv, summary.csv, gates.txt, community.png
"""
from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reproduce_core_result import (
    make_cyprus_style_pv_profile, make_load_profiles, make_batteries, compute_pv_metrics,
)
from experiments.spectral_diagnostics import spectral_diagnostics
from semele.battery import Battery
from semele.controllers import belief_neighborhood_controller
from semele.network import make_small_world
from semele.pricing import pv_shaped_price
from semele.simulator import run_simulation

# --- configuration ------------------------------------------------------
SEED = 42
N_HOUSES = 20
N_STEPS = 24
N_RUNS = 100

RHO_VALUES = [0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
PRICE_RULES = ["fixed", "sdr"]
BATTERY_POWER_KW = [3.0, 5.0, 7.0, 10.0]   # per house
BATTERY_HOURS = 4.0                         # capacity = power * 4 h

FEEDER_LIMIT_PER_HOUSE_KW = 6.0             # import and export
PV_PEAK_PER_HOUSE_KW = 5.5
GRID_PRICE, FEED_IN_PRICE = 80.0, 20.0
LOW_PRICE, HIGH_PRICE = 30.0, 70.0
PRICE_SIGMA, FORECAST_SIGMA = 20.0, 1.0
BELIEF_WEIGHT = 0.5

FEEDER_LIMIT_KW = FEEDER_LIMIT_PER_HOUSE_KW * N_HOUSES
OUT = Path(__file__).resolve().parents[1] / "results" / "community"
OUT.mkdir(parents=True, exist_ok=True)


# --- one run --------------------------------------------------------------
def one_run(run_id: int, rho: float, rule: str, p_kw: float, pv_kw, prices_fixed) -> dict:
    rng = np.random.default_rng(SEED + run_id)          # same world for every condition
    loads = make_load_profiles(rng, N_STEPS, N_HOUSES)
    soc0 = make_batteries(rng, N_HOUSES)                # reuse its random initial SoC
    batteries = [Battery(capacity_kwh=p_kw * BATTERY_HOURS, max_charge_kw=p_kw,
                         max_discharge_kw=p_kw, initial_soc=b.initial_soc) for b in soc0]

    np.random.seed(SEED + run_id)                       # same noise draws for every condition
    res = run_simulation(
        load_profiles_kw=loads, prices=prices_fixed, batteries=batteries,
        controller=functools.partial(belief_neighborhood_controller, belief_weight=BELIEF_WEIGHT),
        network=make_small_world(N_HOUSES, seed=SEED),
        low_threshold=LOW_PRICE, high_threshold=HIGH_PRICE,
        feeder_limit_kw=FEEDER_LIMIT_KW, rho_agents=rho,
        forecast_error_sigma_kw=FORECAST_SIGMA, price_sigma=PRICE_SIGMA,
        pv_kw=pv_kw, price_rule=rule, grid_price=GRID_PRICE, feed_in_price=FEED_IN_PRICE,
    )

    # Same day with no batteries: what the feeder breaks on its own.
    net0 = loads.sum(axis=1) - pv_kw
    viol0 = (net0 > FEEDER_LIMIT_KW) | (net0 < -FEEDER_LIMIT_KW)

    w1, pr, _ = spectral_diagnostics(res.per_battery_power_kw, subtract_mean=False)
    charging = np.maximum(0.0, -res.aggregate_battery_power_kw)
    pv = compute_pv_metrics(loads, pv_kw, res.aggregate_battery_power_kw, 1.0)
    return {
        "run_id": run_id, "rho": rho, "rule": rule, "battery_kw": p_kw,
        "peak_charge_ratio": charging.max() / FEEDER_LIMIT_KW,
        "import_viol": res.import_violations.mean(),
        "export_viol": res.export_violations.mean(),
        "battery_caused_viol": res.feeder_violations.mean() - viol0.mean(),
        "curtailment": pv["pv_curtailment_fraction"],
        "w1": w1, "PR": pr,
        "price_std": res.price.std(),
        # Share of hours where the price jumps by more than half its range: flip-flop.
        "price_flips": np.mean(np.abs(np.diff(res.price)) > 0.5 * (GRID_PRICE - FEED_IN_PRICE)),
    }


# --- gates ----------------------------------------------------------------
def gates(s: pd.DataFrame) -> str:
    lines = []
    hi = s[s.rho == 1.0]
    lines.append("GATE A (step 2): does the SDR feedback damp or amplify, at rho = 1?")
    lines.append("  battery  | peak/limit fixed sdr | battery-caused viol fixed sdr | price flips fixed sdr")
    for p in BATTERY_POWER_KW:
        f = hi[(hi.rule == "fixed") & (hi.battery_kw == p)].iloc[0]
        d = hi[(hi.rule == "sdr") & (hi.battery_kw == p)].iloc[0]
        verdict = ("amplifies" if d.battery_caused_viol > f.battery_caused_viol + 0.01
                   else "damps" if d.battery_caused_viol < f.battery_caused_viol - 0.01
                   else "no clear effect")
        lines.append(f"  {p:4.0f} kW  |  {f.peak_charge_ratio:.3f} {d.peak_charge_ratio:.3f}"
                     f"     |  {f.battery_caused_viol:.3f} {d.battery_caused_viol:.3f}"
                     f"              |  {f.price_flips:.2f} {d.price_flips:.2f}   -> {verdict}")
    lines.append("  price flips > 0.3 = the price is flip-flopping hour to hour (cobweb oscillation).")

    lines.append("\nGATE B (step 3): smallest battery size whose charging alone breaks the feeder (rho = 1)")
    for rule in PRICE_RULES:
        sub = hi[(hi.rule == rule) & (hi.battery_caused_viol > 0.0)].sort_values("battery_kw")
        if sub.empty:
            lines.append(f"  {rule}: none up to {max(BATTERY_POWER_KW):.0f} kW -> risk is theoretical here")
        else:
            r = sub.iloc[0]
            lines.append(f"  {rule}: {r.battery_kw:.0f} kW (battery-caused violation fraction {r.battery_caused_viol:.3f})")
    return "\n".join(lines)


# --- plot -----------------------------------------------------------------
def plot(s: pd.DataFrame) -> None:
    panels = [("peak_charge_ratio", "peak charge / feeder limit"),
              ("battery_caused_viol", "violations caused by batteries"),
              ("curtailment", "PV curtailment fraction"),
              ("w1", "w1 (fleet moving as one)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, label) in zip(axes.flat, panels):
        for p, color in zip(BATTERY_POWER_KW, ["C0", "C1", "C2", "C3"]):
            for rule, ls in [("fixed", "-"), ("sdr", "--")]:
                sub = s[(s.battery_kw == p) & (s.rule == rule)]
                ax.plot(sub.rho, sub[col], ls, color=color, marker="o", ms=3,
                        label=f"{p:.0f} kW {rule}")
        ax.set_xlabel("rho"); ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("PV community: fixed price (solid) vs SDR local price (dashed)")
    fig.tight_layout()
    fig.savefig(OUT / "community.png", dpi=160)


def main() -> None:
    pv_kw = make_cyprus_style_pv_profile(N_STEPS, PV_PEAK_PER_HOUSE_KW * N_HOUSES)
    prices_fixed = pv_shaped_price(pv_kw, high=GRID_PRICE, swing=GRID_PRICE - FEED_IN_PRICE)

    rows = [one_run(r, rho, rule, p, pv_kw, prices_fixed)
            for p in BATTERY_POWER_KW for rule in PRICE_RULES
            for rho in RHO_VALUES for r in range(N_RUNS)]
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw.csv", index=False)

    summary = raw.groupby(["battery_kw", "rule", "rho"], as_index=False).mean(numeric_only=True)
    summary = summary.drop(columns="run_id")
    summary.to_csv(OUT / "summary.csv", index=False)

    text = gates(summary)
    (OUT / "gates.txt").write_text(text + "\n")
    print(text)
    plot(summary)
    print(f"\nsaved results to {OUT}")


if __name__ == "__main__":
    main()
