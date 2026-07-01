import random
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lampyris.battery import Battery
from lampyris.controllers import randomized_tou_controller, tou_controller
from lampyris.simulator import (
    compute_max_ramp_kw,
    compute_peak_kw,
    compute_total_feeder_excess_kw,
    count_feeder_violations,
    run_simulation,
)

np.random.seed(42)
random.seed(42)

n_steps = 24
scenario_specs = [
    {"label": "light", "n_batteries": 5, "feeder_limit_kw": 60.0},
    {"label": "medium", "n_batteries": 10, "feeder_limit_kw": 60.0},
    {"label": "hard", "n_batteries": 20, "feeder_limit_kw": 80.0},
]
randomness_values = [0.2, 0.6, 0.9]

# One shared superset of synthetic data. Each scenario uses a slice so
# controller comparisons stay fair within and across scenarios.
loads_master = np.random.uniform(
    5, 15, size=(n_steps, max(spec["n_batteries"] for spec in scenario_specs))
)
prices = np.linspace(10, 100, n_steps)


def make_batteries(n_batteries: int) -> list[Battery]:
    """Create a fresh battery fleet for each experiment run."""
    return [
        Battery(capacity_kwh=50, max_charge_kw=10, max_discharge_kw=10)
        for _ in range(n_batteries)
    ]


def summarize_results(
    name: str,
    results,
    feeder_limit_kw: float,
    scenario_label: str,
    randomness: float | None,
) -> dict:
    """Print and return the core metrics for one experiment run."""
    metrics = {
        "scenario": scenario_label,
        "controller": name,
        "randomness": randomness,
        "peak": compute_peak_kw(results.aggregate_load_kw),
        "ramp": compute_max_ramp_kw(results.aggregate_load_kw),
        "violations": count_feeder_violations(results.feeder_violations),
        "total_excess": compute_total_feeder_excess_kw(
            results.aggregate_load_kw, feeder_limit_kw
        ),
    }

    print(f"=== {name} ===")
    if randomness is not None:
        print("Randomness:", randomness)
    print("Peak:", metrics["peak"])
    print("Ramp:", metrics["ramp"])
    print("Violations:", metrics["violations"])
    print("Total Excess:", metrics["total_excess"])
    print()

    return metrics


def run_one_experiment(
    name: str,
    controller,
    loads: np.ndarray,
    n_batteries: int,
    feeder_limit_kw: float,
    scenario_label: str,
    randomness: float | None = None,
) -> dict:
    """Run one controller on one scenario, print metrics, and return them."""
    results = run_simulation(
        load_profiles_kw=loads,
        prices=prices,
        batteries=make_batteries(n_batteries),
        controller=controller,
        dt_hours=1.0,
        low_threshold=30,
        high_threshold=70,
        feeder_limit_kw=feeder_limit_kw,
    )

    metrics = summarize_results(
        name, results, feeder_limit_kw, scenario_label, randomness
    )
    metrics["n_batteries"] = n_batteries
    metrics["feeder_limit_kw"] = feeder_limit_kw
    return metrics


all_results: list[dict] = []

for spec in scenario_specs:
    scenario_label = spec["label"]
    n_batteries = spec["n_batteries"]
    feeder_limit_kw = spec["feeder_limit_kw"]
    scenario_loads = loads_master[:, :n_batteries]

    print("=" * 60)
    print(
        f"Scenario: {scenario_label} | n_batteries={n_batteries}, feeder_limit_kw={feeder_limit_kw}"
    )
    print("=" * 60)

    # Baseline for comparison
    all_results.append(
        run_one_experiment(
            "Baseline - TOU",
            tou_controller,
            scenario_loads,
            n_batteries,
            feeder_limit_kw,
            scenario_label,
        )
    )

    # Randomness sweep
    for randomness in randomness_values:
        all_results.append(
            run_one_experiment(
                "Randomized TOU",
                lambda **kwargs: randomized_tou_controller(
                    **kwargs, randomness=randomness
                ),
                scenario_loads,
                n_batteries,
                feeder_limit_kw,
                scenario_label,
                randomness=randomness,
            )
        )

results_df = pd.DataFrame(all_results)
results_df = results_df[
    [
        "scenario",
        "n_batteries",
        "feeder_limit_kw",
        "controller",
        "randomness",
        "peak",
        "ramp",
        "violations",
        "total_excess",
    ]
]

print("=" * 60)
print("Compact results table")
print("=" * 60)
print(results_df.to_string(index=False))

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "randomness_sweep_results.csv"
results_df.to_csv(output_path, index=False)
print()
print(f"Saved results table to: {output_path}")