"""
Generate the paper figures for the Semele note.

Outputs
results/paper/fig1_w1_vs_rho.png
results/paper/fig1_w1_vs_rho.csv
results/paper/fig2_consensus_alignment.png
results/paper/fig2_consensus_alignment.csv
"""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reproduce_core_result import (
    make_cyprus_style_pv_profile,
    make_price_signal,
    make_load_profiles,
    make_batteries,
    build_topologies,
    N_STEPS,
    N_BATTERIES,
    RHO_VALUES,
    DT_HOURS,
    PV_PEAK_KW,
    FEEDER_LIMIT_KW,
    LOW_PRICE,
    HIGH_PRICE,
    FORECAST_ERROR_SIGMA_KW,
    PRICE_SIGMA,
)
from experiments.spectral_diagnostics import spectral_diagnostics
from semele.controllers import belief_neighborhood_controller
from semele.simulator import run_simulation


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PAPER_RESULTS_DIR = RESULTS_DIR / "paper"
PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BELIEF_WEIGHT = 0.5
SUBTRACT_MEAN = False
CONSENSUS_RHOS = [0.0, 0.5, 0.85, 1.0]

FIG1_PNG = PAPER_RESULTS_DIR / "fig1_w1_vs_rho.png"
FIG1_CSV = PAPER_RESULTS_DIR / "fig1_w1_vs_rho.csv"

FIG2_PNG = PAPER_RESULTS_DIR / "fig2_consensus_alignment.png"
FIG2_CSV = PAPER_RESULTS_DIR / "fig2_consensus_alignment.csv"


# Fixed synthetic world used by both figures. Only topology/rho changes.
_PV_KW = make_cyprus_style_pv_profile(n_steps=N_STEPS, pv_peak_kw=PV_PEAK_KW)
_PRICES = make_price_signal(_PV_KW)
_LOAD_PROFILES_KW = make_load_profiles(
    rng=np.random.default_rng(SEED),
    n_steps=N_STEPS,
    n_batteries=N_BATTERIES,
)


def _run_trajectory(topology_factory, rho: float) -> np.ndarray:
    """Run one deterministic trajectory and return per-battery power, shape (T, N)."""
    controller = functools.partial(
        belief_neighborhood_controller,
        belief_weight=BELIEF_WEIGHT,
    )
    batteries = make_batteries(
        rng=np.random.default_rng(SEED),
        n_batteries=N_BATTERIES,
    )
    network = topology_factory()

    # The simulator draws forecast noise via np.random, so seed immediately
    # before the call. This keeps Z/private draws identical across rho/topology;
    # rho controls only the mixing between common and private noise.
    np.random.seed(SEED)

    result = run_simulation(
        load_profiles_kw=_LOAD_PROFILES_KW,
        prices=_PRICES,
        batteries=batteries,
        controller=controller,
        dt_hours=DT_HOURS,
        low_threshold=LOW_PRICE,
        high_threshold=HIGH_PRICE,
        feeder_limit_kw=FEEDER_LIMIT_KW,
        rho_agents=float(rho),
        forecast_error_sigma_kw=FORECAST_ERROR_SIGMA_KW,
        price_sigma=PRICE_SIGMA,
        network=network,
        generation_kw=_PV_KW,
    )
    return result.per_battery_power_kw


def _consensus_alignment(u1: np.ndarray) -> float:
    """Absolute cosine alignment between dominant spatial mode and 1/sqrt(N)."""
    u1 = np.asarray(u1, dtype=float)
    ones = np.ones_like(u1) / np.sqrt(len(u1))
    return float(abs(np.dot(u1, ones)) / (np.linalg.norm(u1) + 1e-12))


def compute_fig1_data() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    topologies = build_topologies()

    for topology_name, topology_factory in topologies.items():
        for rho in RHO_VALUES:
            power_tn = _run_trajectory(topology_factory, float(rho))
            w1, pr, _ = spectral_diagnostics(
                power_tn,
                subtract_mean=SUBTRACT_MEAN,
            )
            rows.append(
                {
                    "topology": topology_name,
                    "rho": float(rho),
                    "w1": float(w1),
                    "participation_ratio": float(pr),
                    "belief_weight": BELIEF_WEIGHT,
                    "seed": SEED,
                    "subtract_mean": SUBTRACT_MEAN,
                }
            )

    return pd.DataFrame(rows).sort_values(["topology", "rho"])


def compute_fig2_data() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    topologies = build_topologies()

    for topology_name, topology_factory in topologies.items():
        for rho in CONSENSUS_RHOS:
            power_tn = _run_trajectory(topology_factory, float(rho))
            w1, pr, u1 = spectral_diagnostics(
                power_tn,
                subtract_mean=SUBTRACT_MEAN,
            )
            rows.append(
                {
                    "topology": topology_name,
                    "rho": float(rho),
                    "w1": float(w1),
                    "participation_ratio": float(pr),
                    "alignment": _consensus_alignment(u1),
                    "belief_weight": BELIEF_WEIGHT,
                    "seed": SEED,
                    "subtract_mean": SUBTRACT_MEAN,
                }
            )

    return pd.DataFrame(rows).sort_values(["topology", "rho"])


def plot_fig1(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    for topology_name, sub in df.groupby("topology", sort=False):
        sub = sub.sort_values("rho")
        ax.plot(sub["rho"], sub["w1"], marker="o", label=topology_name.replace("_", "-"))

    ax.set_xlabel(r"correlation of forecast error $\rho$")
    ax.set_ylabel(r"dominant-mode fraction $w_1$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG1_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))

    for topology_name, sub in df.groupby("topology", sort=False):
        sub = sub.sort_values("rho")
        ax.plot(sub["rho"], sub["alignment"], marker="o", label=topology_name.replace("_", "-"))

    ax.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel(r"correlation of forecast error $\rho$")
    ax.set_ylabel(r"alignment $|\langle u_1, \mathbf{1}\rangle|$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.95, 1.005)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG2_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig1 = compute_fig1_data()
    fig1.to_csv(FIG1_CSV, index=False)
    plot_fig1(fig1)

    fig2 = compute_fig2_data()
    fig2.to_csv(FIG2_CSV, index=False)
    plot_fig2(fig2)

    print("saved paper figures:")
    for path in [FIG1_PNG, FIG1_CSV, FIG2_PNG, FIG2_CSV]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
