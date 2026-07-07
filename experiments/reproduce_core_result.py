"""
Reproduce the core Semele result.

Question:
    When do distributed batteries help PV integration, and when do shared
    forecast/price signals synchronize them into grid-level stress?

Core story:
    Low rho_agents:
        batteries act less synchronously and can absorb PV surplus.

    High rho_agents:
        shared signal exposure synchronizes battery actions, increasing
        aggregate charging peaks and feeder stress.

Outputs:
    results/core_rho_sweep.csv
    results/core_summary.csv
    results/core_summary.json
    results/core_rho_sweep.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semele.battery import Battery
from semele.controllers import belief_neighborhood_controller
from semele.network import Network, make_linear, make_small_world, make_star
from semele.simulator import run_simulation


# ---------------------------------------------------------------------
# Simple user-facing configuration
# ---------------------------------------------------------------------

MASTER_SEED = 42

N_STEPS = 24
N_BATTERIES = 20
N_RUNS = 200

RHO_VALUES = np.linspace(0.0, 1.0, 21)

DT_HOURS = 1.0

BATTERY_CAPACITY_KWH = 20.0
BATTERY_POWER_KW = 5.0

# Scaled system assumptions.
# Change these instead of hardcoding FEEDER_LIMIT_KW or PV_PEAK_KW.
FEEDER_LIMIT_PER_BATTERY_KW = 6.0
PV_PEAK_PER_BATTERY_KW = 5.5

FEEDER_LIMIT_KW = FEEDER_LIMIT_PER_BATTERY_KW * N_BATTERIES
PV_PEAK_KW = PV_PEAK_PER_BATTERY_KW * N_BATTERIES

LOW_PRICE = 30.0
HIGH_PRICE = 70.0

FORECAST_ERROR_SIGMA_KW = 1.0
PRICE_SIGMA = 20.0

NETWORK_ALPHA = 0.1
NETWORK_K = 0.2

TOPOLOGIES = ("linear", "star", "small_world")


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "core_rho_sweep.csv"
SUMMARY_CSV_PATH = RESULTS_DIR / "core_summary.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "core_summary.json"
PLOT_PATH = RESULTS_DIR / "core_rho_sweep.png"


# ---------------------------------------------------------------------
# Synthetic island-grid profiles
# ---------------------------------------------------------------------

def make_cyprus_style_pv_profile(n_steps: int, pv_peak_kw: float) -> np.ndarray:
    """
    Synthetic Cyprus-style PV profile.

    This is not real Cyprus data. It only encodes the qualitative structure:
        - zero PV at night
        - strong midday solar peak
        - mild cloud/noise shape
    """
    hours = np.arange(n_steps)

    daylight_shape = np.exp(-0.5 * ((hours - 12.0) / 3.0) ** 2)
    daylight_shape[hours < 6] = 0.0
    daylight_shape[hours > 18] = 0.0

    pv_kw = pv_peak_kw * daylight_shape

    # Deterministic cloud dip around early afternoon.
    cloud_dip = 1.0 - 0.25 * np.exp(-0.5 * ((hours - 14.0) / 1.2) ** 2)

    return pv_kw * cloud_dip


def make_load_profiles(
    rng: np.random.Generator,
    n_steps: int,
    n_batteries: int,
) -> np.ndarray:
    """
    Synthetic local demand profiles.

    Each battery/customer has a small local load.
    Total load is the sum across agents.
    """
    hours = np.arange(n_steps)

    morning = 1.2 * np.exp(-0.5 * ((hours - 8.0) / 2.0) ** 2)
    evening = 2.0 * np.exp(-0.5 * ((hours - 19.0) / 2.5) ** 2)
    base = 2.2 + morning + evening

    profiles = []
    for _ in range(n_batteries):
        scale = rng.uniform(0.85, 1.15)
        noise = rng.normal(0.0, 0.15, size=n_steps)
        profile = np.maximum(0.5, scale * base + noise)
        profiles.append(profile)

    return np.column_stack(profiles)


def make_price_signal(pv_kw: np.ndarray) -> np.ndarray:
    """
    Simple price/curtailment proxy.

    Price is low when PV is high, encouraging batteries to charge during solar surplus.
    """
    pv_norm = pv_kw / max(float(np.max(pv_kw)), 1e-9)
    prices = 80.0 - 60.0 * pv_norm
    return prices


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------

def make_batteries(
    rng: np.random.Generator,
    n_batteries: int,
) -> list[Battery]:
    """Create a fresh battery fleet for each simulation run."""
    return [
        Battery(
            capacity_kwh=BATTERY_CAPACITY_KWH,
            max_charge_kw=BATTERY_POWER_KW,
            max_discharge_kw=BATTERY_POWER_KW,
            initial_soc=float(rng.uniform(0.2, 0.8)),
        )
        for _ in range(n_batteries)
    ]


def build_topologies() -> dict[str, Callable[[], Network]]:
    """Clean topology set for the reproducible core result."""
    factories: dict[str, Callable[[], Network]] = {
        "linear": lambda: make_linear(
            N_BATTERIES,
            k=NETWORK_K,
            alpha=NETWORK_ALPHA,
        ),
        "star": lambda: make_star(
            N_BATTERIES,
            k=NETWORK_K,
            alpha=NETWORK_ALPHA,
        ),
        "small_world": lambda: make_small_world(
            N_BATTERIES,
            k_nn=4,
            p=0.1,
            k=NETWORK_K,
            alpha=NETWORK_ALPHA,
            seed=MASTER_SEED,
        ),
    }

    return {name: factories[name] for name in TOPOLOGIES}


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def synchronization_index(per_battery_power_kw: np.ndarray) -> float:
    """
    Average cross-agent action spread through time.

    Lower value means agents are acting more similarly.
    """
    return float(np.mean(np.std(per_battery_power_kw, axis=1)))


def peak_charge_kw(aggregate_battery_power_kw: np.ndarray) -> float:
    """
    Maximum aggregate charging demand.

    Sign convention:
        battery power < 0 means charging.

    Therefore aggregate charging is -aggregate_battery_power_kw when negative.
    """
    charging_kw = np.maximum(0.0, -aggregate_battery_power_kw)
    return float(np.max(charging_kw))


def compute_pv_metrics(
    load_profiles_kw: np.ndarray,
    pv_kw: np.ndarray,
    aggregate_battery_power_kw: np.ndarray,
    dt_hours: float,
) -> dict[str, float]:
    """
    Estimate PV absorption and curtailment-risk metrics.

    This is a simple energy-accounting proxy, not a full power-flow model.

    Before batteries:
        surplus PV = max(0, PV - load)

    Battery charging:
        charging_kw = max(0, -battery_power)

    PV absorbed by batteries:
        min(surplus PV, charging_kw)

    Remaining curtailment risk:
        max(0, surplus PV - charging_kw)
    """
    total_load_kw = load_profiles_kw.sum(axis=1)
    surplus_pv_kw = np.maximum(0.0, pv_kw - total_load_kw)
    charging_kw = np.maximum(0.0, -aggregate_battery_power_kw)

    pv_absorbed_kw = np.minimum(surplus_pv_kw, charging_kw)
    curtailed_kw = np.maximum(0.0, surplus_pv_kw - charging_kw)

    surplus_kwh = float(np.sum(surplus_pv_kw) * dt_hours)
    absorbed_kwh = float(np.sum(pv_absorbed_kw) * dt_hours)
    curtailed_kwh = float(np.sum(curtailed_kw) * dt_hours)

    curtailment_fraction = curtailed_kwh / surplus_kwh if surplus_kwh > 0.0 else 0.0
    absorption_fraction = absorbed_kwh / surplus_kwh if surplus_kwh > 0.0 else 0.0

    return {
        "pv_surplus_kwh": surplus_kwh,
        "pv_absorbed_by_batteries_kwh": absorbed_kwh,
        "pv_curtailment_risk_kwh": curtailed_kwh,
        "pv_curtailment_fraction": curtailment_fraction,
        "pv_absorption_fraction": absorption_fraction,
    }


# ---------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------

def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw run results into mean/std curves."""
    return (
        df.groupby(["topology", "rho_agents"], as_index=False)
        .agg(
            sync_mean=("sync_index", "mean"),
            sync_std=("sync_index", "std"),
            violation_mean=("feeder_violation_fraction", "mean"),
            violation_std=("feeder_violation_fraction", "std"),
            peak_charge_ratio_mean=("peak_charge_ratio", "mean"),
            peak_charge_ratio_std=("peak_charge_ratio", "std"),
            curtailment_fraction_mean=("pv_curtailment_fraction", "mean"),
            curtailment_fraction_std=("pv_curtailment_fraction", "std"),
            absorption_fraction_mean=("pv_absorption_fraction", "mean"),
            absorption_fraction_std=("pv_absorption_fraction", "std"),
        )
        .sort_values(["topology", "rho_agents"])
    )


def make_summary_payload(summary: pd.DataFrame) -> dict:
    """Create a small JSON summary for reproducibility and quick inspection."""
    headline_by_topology: dict[str, dict[str, float]] = {}

    for topology_name in summary["topology"].unique():
        sub = summary[summary["topology"] == topology_name].sort_values("rho_agents")

        low = sub.iloc[0]
        high = sub.iloc[-1]

        headline_by_topology[topology_name] = {
            "sync_mean_at_rho_0": float(low["sync_mean"]),
            "sync_mean_at_rho_1": float(high["sync_mean"]),
            "violation_mean_at_rho_0": float(low["violation_mean"]),
            "violation_mean_at_rho_1": float(high["violation_mean"]),
            "peak_charge_ratio_at_rho_0": float(low["peak_charge_ratio_mean"]),
            "peak_charge_ratio_at_rho_1": float(high["peak_charge_ratio_mean"]),
            "curtailment_fraction_at_rho_0": float(low["curtailment_fraction_mean"]),
            "curtailment_fraction_at_rho_1": float(high["curtailment_fraction_mean"]),
        }

    return {
        "project": "Semele",
        "question": (
            "When do distributed batteries help PV integration, and when do "
            "shared signals synchronize them into grid-level stress?"
        ),
        "headline_result": (
            "Increasing rho_agents reduces cross-agent action spread, increases "
            "aggregate charging pressure, and can raise feeder-stress and "
            "synthetic PV-curtailment-risk proxies."
        ),
        "important_limitation": (
            "PV curtailment is represented as a simple energy-accounting risk proxy, "
            "not a full power-flow or market-dispatch model."
        ),
        "configuration": {
            "master_seed": MASTER_SEED,
            "n_steps": N_STEPS,
            "n_batteries": N_BATTERIES,
            "n_runs": N_RUNS,
            "rho_values": [float(x) for x in RHO_VALUES],
            "topologies": list(TOPOLOGIES),
            "dt_hours": DT_HOURS,
            "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
            "battery_power_kw": BATTERY_POWER_KW,
            "feeder_limit_kw": FEEDER_LIMIT_KW,
            "feeder_limit_per_battery_kw": FEEDER_LIMIT_PER_BATTERY_KW,
            "pv_peak_kw": PV_PEAK_KW,
            "pv_peak_per_battery_kw": PV_PEAK_PER_BATTERY_KW,
            "low_price": LOW_PRICE,
            "high_price": HIGH_PRICE,
            "forecast_error_sigma_kw": FORECAST_ERROR_SIGMA_KW,
            "price_sigma": PRICE_SIGMA,
            "network_alpha": NETWORK_ALPHA,
            "network_k": NETWORK_K,
        },
        "headline_by_topology": headline_by_topology,
        "outputs": {
            "raw_csv": str(CSV_PATH),
            "summary_csv": str(SUMMARY_CSV_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
            "plot": str(PLOT_PATH),
        },
    }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def make_plot(summary: pd.DataFrame) -> None:
    """
    Save a compact reproduction plot.

    Three panels:
        1. synchronization spread
        2. peak charging pressure
        3. PV curtailment risk
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for topology_name in summary["topology"].unique():
        sub = summary[summary["topology"] == topology_name]

        axes[0].plot(
            sub["rho_agents"],
            sub["sync_mean"],
            marker="o",
            label=topology_name,
        )

        axes[1].plot(
            sub["rho_agents"],
            sub["peak_charge_ratio_mean"],
            marker="o",
            label=topology_name,
        )

        axes[2].plot(
            sub["rho_agents"],
            sub["curtailment_fraction_mean"],
            marker="o",
            label=topology_name,
        )

    axes[0].set_xlabel("rho_agents")
    axes[0].set_ylabel("sync_index")
    axes[0].set_title("Action synchronization")

    axes[1].set_xlabel("rho_agents")
    axes[1].set_ylabel("peak charge / feeder limit")
    axes[1].set_title("Aggregate charging pressure")

    axes[2].set_xlabel("rho_agents")
    axes[2].set_ylabel("PV curtailment-risk fraction")
    axes[2].set_title("PV curtailment risk")

    axes[0].legend()

    fig.suptitle("Semele core result: shared signals, storage coordination, PV risk")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    print(f"saved {PLOT_PATH}")


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    master_rng = np.random.default_rng(MASTER_SEED)

    pv_kw = make_cyprus_style_pv_profile(
        n_steps=N_STEPS,
        pv_peak_kw=PV_PEAK_KW,
    )
    prices = make_price_signal(pv_kw)
    topology_factories = build_topologies()

    rows: list[dict[str, float | int | str]] = []

    for topology_name, topology_factory in topology_factories.items():
        for rho_agents in RHO_VALUES:
            for run_id in range(N_RUNS):
                run_seed = int(master_rng.integers(0, 2**32 - 1))
                run_rng = np.random.default_rng(run_seed)

                # run_simulation currently uses np.random internally for correlated noise.
                # We seed it explicitly so every run is reproducible.
                np.random.seed(run_seed)

                load_profiles_kw = make_load_profiles(
                    rng=run_rng,
                    n_steps=N_STEPS,
                    n_batteries=N_BATTERIES,
                )

                batteries = make_batteries(
                    rng=run_rng,
                    n_batteries=N_BATTERIES,
                )

                network = topology_factory()

                result = run_simulation(
                    load_profiles_kw=load_profiles_kw,
                    prices=prices,
                    batteries=batteries,
                    controller=belief_neighborhood_controller,
                    dt_hours=DT_HOURS,
                    low_threshold=LOW_PRICE,
                    high_threshold=HIGH_PRICE,
                    feeder_limit_kw=FEEDER_LIMIT_KW,
                    rho_agents=float(rho_agents),
                    forecast_error_sigma_kw=FORECAST_ERROR_SIGMA_KW,
                    price_sigma=PRICE_SIGMA,
                    network=network,
                    generation_kw=pv_kw,
                )

                sync = synchronization_index(result.per_battery_power_kw)
                violation_fraction = float(np.mean(result.feeder_violations))
                peak_charge = peak_charge_kw(result.aggregate_battery_power_kw)
                peak_charge_ratio = peak_charge / FEEDER_LIMIT_KW

                pv_metrics = compute_pv_metrics(
                    load_profiles_kw=load_profiles_kw,
                    pv_kw=pv_kw,
                    aggregate_battery_power_kw=result.aggregate_battery_power_kw,
                    dt_hours=DT_HOURS,
                )

                rows.append(
                    {
                        "topology": topology_name,
                        "rho_agents": float(rho_agents),
                        "run_id": run_id,
                        "run_seed": run_seed,
                        "sync_index": sync,
                        "feeder_violation_fraction": violation_fraction,
                        "peak_charge_kw": peak_charge,
                        "peak_charge_ratio": peak_charge_ratio,
                        **pv_metrics,
                    }
                )

            print(f"finished topology={topology_name:12s} rho={rho_agents:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"saved {CSV_PATH}")

    summary = summarize_results(df)
    summary.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"saved {SUMMARY_CSV_PATH}")

    payload = make_summary_payload(summary)
    SUMMARY_JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"saved {SUMMARY_JSON_PATH}")

    make_plot(summary)

    print("done")


if __name__ == "__main__":
    main()