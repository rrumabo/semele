"""
Phase 10 — Mechanism diagnostics.

Question:
Which intermediate variable best predicts feeder violations under correlated belief?

Hypothesis:
rho_agents is the upstream cause, but feeder failure is better explained by
synchronized aggregate charging pressure, especially peak_charge_ratio.

This script records mechanism variables:
- sync_metric
- violation_fraction
- peak_charge_kw
- peak_charge_ratio
- fraction_charging_mean
- fraction_charging_peak
- max_feeder_loading_ratio
- mean_feeder_loading_ratio
- mean_headroom_kw
- min_headroom_kw
- mean_abs_aggregate_power_kw

Outputs:
- results/mechanism_diagnostics_results.csv
- results/mechanism_diagnostics_summary.csv
- results/mechanism_predictor_correlations.csv
- results/mechanism_peak_charge_vs_violation.png
- results/mechanism_sync_vs_violation.png
- results/mechanism_fraction_charging_vs_violation.png
"""

from pathlib import Path
import inspect
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lampyris.battery import Battery
from lampyris.controllers import belief_neighborhood_controller
from lampyris.network import make_linear
from lampyris.simulator import run_simulation


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RESULTS_DIR / "mechanism_diagnostics_results.csv"
SUMMARY_PATH = RESULTS_DIR / "mechanism_diagnostics_summary.csv"
CORR_PATH = RESULTS_DIR / "mechanism_predictor_correlations.csv"

PLOT_PEAK_CHARGE_PATH = RESULTS_DIR / "mechanism_peak_charge_vs_violation.png"
PLOT_SYNC_PATH = RESULTS_DIR / "mechanism_sync_vs_violation.png"
PLOT_CHARGING_FRACTION_PATH = RESULTS_DIR / "mechanism_fraction_charging_vs_violation.png"

fleet_sizes = [10, 30, 50]
rho_values = np.linspace(0.0, 1.0, 20)
n_runs = 20
n_steps = 24

price_sigma = 20.0
forecast_error_sigma_kw = 1.0

battery_capacity_kwh = 10.0
max_power_kw = 5.0
network_k = 0.35
alpha = 0.30


def make_loads_and_prices(rng: np.random.Generator, n_batteries: int):
    load_profiles_kw = rng.uniform(1.0, 5.0, size=(n_steps, n_batteries))
    prices = rng.uniform(10.0, 100.0, size=n_steps)
    return load_profiles_kw, prices


def make_batteries(rng: np.random.Generator, n_batteries: int):
    signature = inspect.signature(Battery)
    battery_kwargs = {}

    if "capacity_kwh" in signature.parameters:
        battery_kwargs["capacity_kwh"] = battery_capacity_kwh
    elif "capacity" in signature.parameters:
        battery_kwargs["capacity"] = battery_capacity_kwh

    if "max_power_kw" in signature.parameters:
        battery_kwargs["max_power_kw"] = max_power_kw
    elif "power_kw" in signature.parameters:
        battery_kwargs["power_kw"] = max_power_kw
    elif "p_max_kw" in signature.parameters:
        battery_kwargs["p_max_kw"] = max_power_kw

    if "max_charge_kw" in signature.parameters:
        battery_kwargs["max_charge_kw"] = max_power_kw
    if "max_discharge_kw" in signature.parameters:
        battery_kwargs["max_discharge_kw"] = max_power_kw

    batteries = []
    initial_socs = []

    for _ in range(n_batteries):
        kwargs = dict(battery_kwargs)
        soc = float(rng.uniform(0.2, 0.8))
        initial_socs.append(soc)

        if "initial_soc" in signature.parameters:
            kwargs["initial_soc"] = soc
        elif "soc" in signature.parameters:
            kwargs["soc"] = soc
        elif "initial_soc_kwh" in signature.parameters:
            kwargs["initial_soc_kwh"] = soc * battery_capacity_kwh

        batteries.append(Battery(**kwargs))

    return batteries, np.asarray(initial_socs)


def compute_sync_metric(per_battery_power: np.ndarray) -> float:
    return float(np.mean(np.std(per_battery_power, axis=1)))


def compute_feeder_load_kw(result, load_profiles_kw: np.ndarray, aggregate_battery_power: np.ndarray):
    if hasattr(result, "feeder_load_kw"):
        return np.asarray(result.feeder_load_kw)
    if hasattr(result, "aggregate_load_kw"):
        return np.asarray(result.aggregate_load_kw)

    base_load_kw = np.sum(load_profiles_kw, axis=1)

    # Convention from the previous sweeps:
    # negative battery power means charging, so feeder load increases by -P_batt.
    return base_load_kw - aggregate_battery_power


def compute_violation_fraction(feeder_load_kw: np.ndarray, feeder_limit_kw: float) -> float:
    return float(np.mean(feeder_load_kw > feeder_limit_kw))


def main():
    records = []

    for n_batteries in fleet_sizes:
        feeder_limit_kw = 6.0 * n_batteries
        network = make_linear(n_batteries, k=network_k, alpha=alpha)

        for rho in rho_values:
            for run in range(n_runs):
                seed = 20_000 + 1_000 * n_batteries + 100 * run + int(round(rho * 1_000))
                rng = np.random.default_rng(seed)

                load_profiles_kw, prices = make_loads_and_prices(rng, n_batteries)
                batteries, initial_socs = make_batteries(rng, n_batteries)

                result = run_simulation(
                    batteries=batteries,
                    prices=prices,
                    load_profiles_kw=load_profiles_kw,
                    controller=belief_neighborhood_controller,
                    rho_agents=float(rho),
                    forecast_error_sigma_kw=forecast_error_sigma_kw,
                    price_sigma=price_sigma,
                    feeder_limit_kw=feeder_limit_kw,
                    network=network,
                )

                per_battery_power = np.asarray(result.per_battery_power_kw)
                aggregate_battery_power = per_battery_power.sum(axis=1)

                feeder_load_kw = compute_feeder_load_kw(
                    result,
                    load_profiles_kw,
                    aggregate_battery_power,
                )

                base_load_kw = np.sum(load_profiles_kw, axis=1)
                headroom_before_battery_kw = feeder_limit_kw - base_load_kw
                headroom_after_battery_kw = feeder_limit_kw - feeder_load_kw

                charging_mask = per_battery_power < 0.0
                discharging_mask = per_battery_power > 0.0

                fraction_charging_t = charging_mask.mean(axis=1)
                fraction_discharging_t = discharging_mask.mean(axis=1)

                charge_power_t = np.maximum(-aggregate_battery_power, 0.0)
                discharge_power_t = np.maximum(aggregate_battery_power, 0.0)

                peak_charge_kw = float(np.max(charge_power_t))
                peak_discharge_kw = float(np.max(discharge_power_t))

                records.append(
                    {
                        "n_batteries": n_batteries,
                        "rho_agents": float(rho),
                        "run": run,
                        "feeder_limit_kw": feeder_limit_kw,
                        "sync_metric": compute_sync_metric(per_battery_power),
                        "violation_fraction": compute_violation_fraction(
                            feeder_load_kw,
                            feeder_limit_kw,
                        ),
                        "peak_charge_kw": peak_charge_kw,
                        "peak_charge_ratio": peak_charge_kw / feeder_limit_kw,
                        "peak_discharge_kw": peak_discharge_kw,
                        "peak_discharge_ratio": peak_discharge_kw / feeder_limit_kw,
                        "mean_abs_aggregate_power_kw": float(
                            np.mean(np.abs(aggregate_battery_power))
                        ),
                        "mean_abs_aggregate_power_ratio": float(
                            np.mean(np.abs(aggregate_battery_power)) / feeder_limit_kw
                        ),
                        "fraction_charging_mean": float(np.mean(fraction_charging_t)),
                        "fraction_charging_peak": float(np.max(fraction_charging_t)),
                        "fraction_discharging_mean": float(np.mean(fraction_discharging_t)),
                        "fraction_discharging_peak": float(np.max(fraction_discharging_t)),
                        "max_feeder_loading_ratio": float(
                            np.max(feeder_load_kw / feeder_limit_kw)
                        ),
                        "mean_feeder_loading_ratio": float(
                            np.mean(feeder_load_kw / feeder_limit_kw)
                        ),
                        "mean_headroom_before_battery_kw": float(
                            np.mean(headroom_before_battery_kw)
                        ),
                        "min_headroom_before_battery_kw": float(
                            np.min(headroom_before_battery_kw)
                        ),
                        "mean_headroom_after_battery_kw": float(
                            np.mean(headroom_after_battery_kw)
                        ),
                        "min_headroom_after_battery_kw": float(
                            np.min(headroom_after_battery_kw)
                        ),
                        "mean_initial_soc": float(np.mean(initial_socs)),
                        "low_soc_fraction": float(np.mean(initial_socs < 0.35)),
                        "high_soc_fraction": float(np.mean(initial_socs > 0.65)),
                    }
                )

            print(f"finished n={n_batteries:<3} rho={rho:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print(f"saved {CSV_PATH}")

    summary = (
        df.groupby(["n_batteries", "rho_agents"], as_index=False)
        .agg(
            sync_mean=("sync_metric", "mean"),
            violation_mean=("violation_fraction", "mean"),
            peak_charge_ratio_mean=("peak_charge_ratio", "mean"),
            peak_charge_ratio_std=("peak_charge_ratio", "std"),
            fraction_charging_mean=("fraction_charging_mean", "mean"),
            fraction_charging_peak_mean=("fraction_charging_peak", "mean"),
            max_feeder_loading_ratio_mean=("max_feeder_loading_ratio", "mean"),
            mean_feeder_loading_ratio_mean=("mean_feeder_loading_ratio", "mean"),
            mean_headroom_before_battery_kw=("mean_headroom_before_battery_kw", "mean"),
            min_headroom_before_battery_kw=("min_headroom_before_battery_kw", "mean"),
            mean_initial_soc=("mean_initial_soc", "mean"),
            low_soc_fraction=("low_soc_fraction", "mean"),
            high_soc_fraction=("high_soc_fraction", "mean"),
        )
    )

    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"saved {SUMMARY_PATH}")

    predictor_columns = [
        "rho_agents",
        "sync_metric",
        "peak_charge_ratio",
        "mean_abs_aggregate_power_ratio",
        "fraction_charging_mean",
        "fraction_charging_peak",
        "max_feeder_loading_ratio",
        "mean_feeder_loading_ratio",
        "mean_headroom_before_battery_kw",
        "min_headroom_before_battery_kw",
        "mean_initial_soc",
        "low_soc_fraction",
        "high_soc_fraction",
    ]

    corr_records = []
    for n_batteries, group in df.groupby("n_batteries"):
        for col in predictor_columns:
            corr = group[[col, "violation_fraction"]].corr().iloc[0, 1]
            corr_records.append(
                {
                    "n_batteries": int(n_batteries),
                    "predictor": col,
                    "corr_with_violation_fraction": float(corr),
                    "abs_corr": float(abs(corr)),
                }
            )

    corr_df = pd.DataFrame(corr_records).sort_values(
        ["n_batteries", "abs_corr"],
        ascending=[True, False],
    )
    corr_df.to_csv(CORR_PATH, index=False)
    print(f"saved {CORR_PATH}")

    print("\nTop predictors by fleet size")
    for n_batteries, group in corr_df.groupby("n_batteries"):
        print(f"\nN = {n_batteries}")
        print(
            group.head(8)[
                ["predictor", "corr_with_violation_fraction"]
            ].to_string(index=False)
        )

    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["peak_charge_ratio"],
        df["violation_fraction"],
        alpha=0.45,
    )
    plt.xlabel("Peak charge ratio")
    plt.ylabel("Violation fraction")
    plt.title("Mechanism: peak charge ratio vs feeder violations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PEAK_CHARGE_PATH, dpi=200)
    print(f"saved {PLOT_PEAK_CHARGE_PATH}")

    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["sync_metric"],
        df["violation_fraction"],
        alpha=0.45,
    )
    plt.xlabel("Sync metric / action dispersion")
    plt.ylabel("Violation fraction")
    plt.title("Mechanism: action dispersion vs feeder violations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_SYNC_PATH, dpi=200)
    print(f"saved {PLOT_SYNC_PATH}")

    plt.figure(figsize=(8, 5))
    plt.scatter(
        df["fraction_charging_peak"],
        df["violation_fraction"],
        alpha=0.45,
    )
    plt.xlabel("Peak fraction of agents charging")
    plt.ylabel("Violation fraction")
    plt.title("Mechanism: simultaneous charging vs feeder violations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_CHARGING_FRACTION_PATH, dpi=200)
    print(f"saved {PLOT_CHARGING_FRACTION_PATH}")


if __name__ == "__main__":
    main()
