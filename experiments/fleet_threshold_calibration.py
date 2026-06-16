"""
Phase 9 — Failure threshold calibration.

Question:
Which feeder-failure threshold definition is comparable across fleet sizes?

Why:
The old baseline-relative 2σ detector is not comparable across N because
baseline variance shrinks with fleet size. This can create artificial early
rho_c_failure estimates and negative protection gaps.

This script re-analyzes results/fleet_size_summary.csv without rerunning
the simulation.

Outputs:
- results/fleet_threshold_calibration.csv
- results/fleet_violation_amplification.csv
- results/fleet_failure_absolute_thresholds.png
- results/fleet_violation_amplification.png
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "fleet_size_summary.csv"

CALIBRATION_PATH = RESULTS_DIR / "fleet_threshold_calibration.csv"
AMPLIFICATION_PATH = RESULTS_DIR / "fleet_violation_amplification.csv"
ABSOLUTE_PLOT_PATH = RESULTS_DIR / "fleet_failure_absolute_thresholds.png"
AMPLIFICATION_PLOT_PATH = RESULTS_DIR / "fleet_violation_amplification.png"

ABSOLUTE_THRESHOLDS = [0.05, 0.10, 0.15]
ABSOLUTE_LIFTS = [0.03, 0.05, 0.10]
RATIO_LEVELS = [2.0, 4.0, 8.0]


def first_rho_where(group: pd.DataFrame, condition) -> float | None:
    rows = group.sort_values("rho_agents")
    hit = rows[condition(rows)]
    if hit.empty:
        return None
    return float(hit.iloc[0]["rho_agents"])


def nearest_rho(group: pd.DataFrame, target: float) -> pd.Series:
    index = (group["rho_agents"] - target).abs().idxmin()
    return group.loc[index]


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. Run experiments/fleet_size_sweep.py first."
        )

    summary = pd.read_csv(SUMMARY_PATH)
    threshold_records = []
    amplification_records = []

    for n_batteries, group in summary.groupby("n_batteries"):
        group = group.sort_values("rho_agents").copy()
        baseline = group[group["rho_agents"] == 0.0].iloc[0]
        baseline_violation = float(baseline["violation_mean"])

        high_rho_row = nearest_rho(group, 0.90)
        final_row = nearest_rho(group, 1.00)

        high_rho_violation = float(high_rho_row["violation_mean"])
        final_violation = float(final_row["violation_mean"])

        eps = 1e-12
        high_rho_ratio = high_rho_violation / max(baseline_violation, eps)
        final_ratio = final_violation / max(baseline_violation, eps)

        amplification_records.append(
            {
                "n_batteries": int(n_batteries),
                "baseline_violation_mean": baseline_violation,
                "rho_near_0_90": float(high_rho_row["rho_agents"]),
                "violation_near_0_90": high_rho_violation,
                "amplification_near_0_90": high_rho_ratio,
                "rho_near_1_00": float(final_row["rho_agents"]),
                "violation_near_1_00": final_violation,
                "amplification_near_1_00": final_ratio,
            }
        )

        record = {
            "n_batteries": int(n_batteries),
            "baseline_violation_mean": baseline_violation,
        }

        for threshold in ABSOLUTE_THRESHOLDS:
            rho_c = first_rho_where(
                group,
                lambda rows, threshold=threshold: rows["violation_mean"] > threshold,
            )
            record[f"rho_c_abs_violation_gt_{threshold:.2f}"] = rho_c

        for lift in ABSOLUTE_LIFTS:
            target = baseline_violation + lift
            rho_c = first_rho_where(
                group,
                lambda rows, target=target: rows["violation_mean"] > target,
            )
            record[f"rho_c_abs_lift_gt_{lift:.2f}"] = rho_c

        for ratio in RATIO_LEVELS:
            target = baseline_violation * ratio
            rho_c = first_rho_where(
                group,
                lambda rows, target=target: rows["violation_mean"] > target,
            )
            record[f"rho_c_ratio_gt_{ratio:.1f}x"] = rho_c

        threshold_records.append(record)

    calibration = pd.DataFrame(threshold_records)
    amplification = pd.DataFrame(amplification_records)

    calibration.to_csv(CALIBRATION_PATH, index=False)
    amplification.to_csv(AMPLIFICATION_PATH, index=False)

    print(f"saved {CALIBRATION_PATH}")
    print(f"saved {AMPLIFICATION_PATH}")

    print("\nFailure threshold calibration")
    print(calibration.to_string(index=False))

    print("\nViolation amplification")
    print(amplification.to_string(index=False))

    plt.figure(figsize=(8, 5))
    for threshold in ABSOLUTE_THRESHOLDS:
        column = f"rho_c_abs_violation_gt_{threshold:.2f}"
        plt.plot(
            calibration["n_batteries"],
            calibration[column],
            marker="o",
            label=f"violation > {threshold:.2f}",
        )

    plt.xlabel("Fleet size N")
    plt.ylabel("Critical rho")
    plt.title("Absolute feeder-failure thresholds vs fleet size")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ABSOLUTE_PLOT_PATH, dpi=200)
    print(f"saved {ABSOLUTE_PLOT_PATH}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        amplification["n_batteries"],
        amplification["amplification_near_0_90"],
        marker="o",
        label="rho ≈ 0.90",
    )
    plt.plot(
        amplification["n_batteries"],
        amplification["amplification_near_1_00"],
        marker="o",
        label="rho = 1.00",
    )
    plt.xlabel("Fleet size N")
    plt.ylabel("Violation amplification ratio")
    plt.title("Relative feeder impact of correlated belief")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(AMPLIFICATION_PLOT_PATH, dpi=200)
    print(f"saved {AMPLIFICATION_PLOT_PATH}")


if __name__ == "__main__":
    main()
