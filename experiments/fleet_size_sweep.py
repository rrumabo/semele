"""
Phase 8 — Fleet size sweep.

Question:
Does the protection window between synchronization collapse and feeder failure
shrink as the number of decentralized battery agents grows?

Design:
- Fixed topology: linear
- Fleet sizes: 5, 10, 20, 30, 50
- Feeder limit scales with fleet size: feeder_limit_kw = 6.0 * n_batteries
- Controller: belief_neighborhood_controller
- Corrupted signal: price
- price_sigma = 20.0
"""

from __future__ import annotations

from pathlib import Path
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


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_PATH = RESULTS_DIR / "fleet_size_results.csv"
SUMMARY_PATH = RESULTS_DIR / "fleet_size_summary.csv"
THRESHOLD_PATH = RESULTS_DIR / "fleet_size_thresholds.csv"
PLOT_PATH = RESULTS_DIR / "fleet_size_thresholds.png"
GAP_PLOT_PATH = RESULTS_DIR / "fleet_size_protection_gap.png"

fleet_sizes = [5, 10, 20, 30, 50]
rho_values = np.linspace(0.0, 1.0, 20)
n_runs = 20
n_steps = 24

price_sigma = 20.0
forecast_error_sigma_kw = 1.0

battery_capacity_kwh = 10.0
max_power_kw = 5.0
battery_power_kw = max_power_kw
network_k = 0.35
alpha = 0.30

def make_loads_and_prices(
    rng: np.random.Generator, n_batteries: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create randomized load and price profiles that excite the controller."""
    load_profiles_kw = rng.uniform(1.0, 5.0, size=(n_steps, n_batteries))
    prices = rng.uniform(10.0, 100.0, size=n_steps)
    return load_profiles_kw, prices


def make_batteries(rng: np.random.Generator, n_batteries: int) -> list[Battery]:
    """Create a fresh heterogeneous battery fleet for each run."""
    initial_socs = rng.uniform(0.2, 0.8, size=n_batteries)
    return [
        Battery(
            capacity_kwh=battery_capacity_kwh,
            max_charge_kw=battery_power_kw,
            max_discharge_kw=battery_power_kw,
            initial_soc=float(initial_soc),
        )
        for initial_soc in initial_socs
    ]


def synchronization_index(per_battery_power_kw: np.ndarray) -> float:
    """Average cross-agent action spread through time."""
    return float(np.mean(np.std(per_battery_power_kw, axis=1)))


def find_rho_c_lower(
    rho_values_in: np.ndarray,
    metric_means: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
) -> float | None:
    """First rho where metric falls below rho=0 baseline by two std."""
    threshold = baseline_mean - 2.0 * baseline_std
    below = np.where(metric_means < threshold)[0]
    return float(rho_values_in[below[0]]) if len(below) > 0 else None


def find_rho_c_upper(
    rho_values_in: np.ndarray,
    metric_means: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
) -> float | None:
    """First rho where metric exceeds rho=0 baseline by two std."""
    threshold = baseline_mean + 2.0 * baseline_std
    above = np.where(metric_means > threshold)[0]
    return float(rho_values_in[above[0]]) if len(above) > 0 else None


def compute_threshold_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute rho_c metrics and fleet-size protection gap."""
    rows: list[dict[str, float | int | None]] = []

    for n_batteries, sub in summary.groupby("n_batteries", sort=False):
        sub = sub.sort_values("rho_agents")

        rho = sub["rho_agents"].to_numpy(dtype=float)
        sync_means = sub["sync_mean"].to_numpy(dtype=float)
        violation_means = sub["violation_mean"].to_numpy(dtype=float)

        baseline = sub[sub["rho_agents"] == 0.0]
        if baseline.empty:
            raise ValueError(f"missing rho=0 baseline for n_batteries={n_batteries}")

        sync_baseline_mean = float(baseline["sync_mean"].iloc[0])
        sync_baseline_std = float(baseline["sync_std"].fillna(0.0).iloc[0])
        violation_baseline_mean = float(baseline["violation_mean"].iloc[0])
        violation_baseline_std = float(baseline["violation_std"].fillna(0.0).iloc[0])

        sync_threshold = sync_baseline_mean - 2.0 * sync_baseline_std
        failure_threshold = violation_baseline_mean + 2.0 * violation_baseline_std

        rho_c_sync_collapse = find_rho_c_lower(
            rho,
            sync_means,
            sync_baseline_mean,
            sync_baseline_std,
        )
        rho_c_failure = find_rho_c_upper(
            rho,
            violation_means,
            violation_baseline_mean,
            violation_baseline_std,
        )

        protection_gap = (
            rho_c_failure - rho_c_sync_collapse
            if rho_c_failure is not None and rho_c_sync_collapse is not None
            else None
        )

        print(
            "thresholds "
            f"n={n_batteries:<3} "
            f"sync_collapse={sync_threshold:.6f} "
            f"failure={failure_threshold:.6f} "
            f"rho_c_sync_collapse={rho_c_sync_collapse} "
            f"rho_c_failure={rho_c_failure}"
        )

        rows.append(
            {
                "n_batteries": n_batteries,
                "feeder_limit_kw": 6.0 * n_batteries,
                "baseline_sync_mean": sync_baseline_mean,
                "baseline_sync_std": sync_baseline_std,
                "baseline_violation_mean": violation_baseline_mean,
                "baseline_violation_std": violation_baseline_std,
                "sync_collapse_threshold": sync_threshold,
                "failure_threshold": failure_threshold,
                "rho_c_sync_collapse": rho_c_sync_collapse,
                "rho_c_failure": rho_c_failure,
                "protection_gap": protection_gap,
            }
        )

    return pd.DataFrame(rows)


def main():
    records = []

    for n_batteries in fleet_sizes:
        feeder_limit_kw = 6.0 * n_batteries
        network = make_linear(n_batteries, k=network_k, alpha=alpha)

        for rho in rho_values:
            for run in range(n_runs):
                seed = 10_000 + 1_000 * n_batteries + 100 * run + int(round(rho * 1_000))
                rng = np.random.default_rng(seed)
                np.random.seed(seed)

                load_profiles_kw, prices = make_loads_and_prices(rng, n_batteries)
                batteries = make_batteries(rng, n_batteries)

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
                aggregate_battery_power = np.asarray(result.aggregate_battery_power_kw)

                records.append(
                    {
                        "n_batteries": n_batteries,
                        "rho_agents": float(rho),
                        "run": run,
                        "feeder_limit_kw": feeder_limit_kw,
                        "sync_index": synchronization_index(per_battery_power),
                        "feeder_violation_fraction": float(
                            np.mean(result.feeder_violations)
                        ),
                        "mean_abs_aggregate_battery_power_kw": float(
                            np.mean(np.abs(aggregate_battery_power))
                        ),
                        "peak_charge_kw": float(np.max(-aggregate_battery_power)),
                        "peak_discharge_kw": float(np.max(aggregate_battery_power)),
                    }
                )

            print(f"finished n={n_batteries:<3} rho={rho:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(CSV_PATH, index=False)
    print(f"saved {CSV_PATH}")

    summary = (
        df.groupby(["n_batteries", "rho_agents"], as_index=False)
        .agg(
            sync_mean=("sync_index", "mean"),
            sync_std=("sync_index", "std"),
            violation_mean=("feeder_violation_fraction", "mean"),
            violation_std=("feeder_violation_fraction", "std"),
            mean_abs_aggregate_power_kw=("mean_abs_aggregate_battery_power_kw", "mean"),
            peak_charge_kw=("peak_charge_kw", "mean"),
            peak_discharge_kw=("peak_discharge_kw", "mean"),
        )
        .sort_values(["n_batteries", "rho_agents"])
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"saved {SUMMARY_PATH}")

    thresholds = compute_threshold_summary(summary)
    thresholds.to_csv(THRESHOLD_PATH, index=False)
    print(f"saved {THRESHOLD_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        thresholds["n_batteries"],
        thresholds["rho_c_sync_collapse"],
        marker="o",
        label="rho_c_sync_collapse",
    )
    plt.plot(
        thresholds["n_batteries"],
        thresholds["rho_c_failure"],
        marker="o",
        label="rho_c_failure",
    )
    plt.xlabel("Fleet size N")
    plt.ylabel("Critical rho")
    plt.title("Critical rho vs fleet size")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    print(f"saved {PLOT_PATH}")

    finite_gap = thresholds.dropna(subset=["protection_gap"])
    if not finite_gap.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(
            finite_gap["n_batteries"],
            finite_gap["protection_gap"],
            marker="o",
        )
        plt.xlabel("Fleet size N")
        plt.ylabel("Protection gap")
        plt.title("Protection window vs fleet size")
        plt.ylim(bottom=0.0)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(GAP_PLOT_PATH, dpi=200)
        print(f"saved {GAP_PLOT_PATH}")
    else:
        print("no finite protection gaps to plot")


if __name__ == "__main__":
    main()
