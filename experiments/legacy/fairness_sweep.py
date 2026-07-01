import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lampyris.battery import Battery
from lampyris.controllers import soft_capped_tou_controller
from lampyris.simulator import (
    compute_peak_kw,
    compute_total_feeder_excess_kw,
    count_feeder_violations,
    run_simulation,
)

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

n_steps     = 24
n_batteries = 10

# Batteries evenly spread: 0.1 (near source) → 1.0 (end of line)
positions = [round((i + 1) / n_batteries, 1) for i in range(n_batteries)]

scenario_specs = [
    {"label": "medium", "feeder_limit_kw": 60.0},
    {"label": "hard",   "feeder_limit_kw": 40.0},
]

# Base loads are LOW — feeder is lightly loaded by background demand.
# Charging is what stresses the feeder, so position effects are visible.
loads_master = np.random.uniform(1, 5, size=(n_steps, n_batteries))
prices       = np.random.uniform(10, 100, n_steps)


def make_batteries(n: int) -> list[Battery]:
    return [
        Battery(capacity_kwh=50, max_charge_kw=10, max_discharge_kw=10,
                initial_soc=random.uniform(0.2, 0.8))
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

all_results = []

for spec in scenario_specs:
    label = spec["label"]
    limit = spec["feeder_limit_kw"]

    result = run_simulation(
        load_profiles_kw = loads_master,
        prices           = prices,
        batteries        = make_batteries(n_batteries),
        controller       = soft_capped_tou_controller,
        dt_hours         = 1.0,
        low_threshold    = 30,
        high_threshold   = 70,
        feeder_limit_kw  = limit,
        positions        = positions,
        position_alpha   = 0.5,
    )

    total_curtailed = result.per_battery_curtailment_kwh.sum()

    print(f"\n{'='*66}")
    print(f"Scenario: {label}  |  feeder_limit={limit} kW  |  "
          f"total curtailed={total_curtailed:.2f} kWh")
    print(f"{'='*66}")
    print(f"  {'bat':<4} {'pos':>6} {'requested':>11} "
          f"{'curtailed':>11} {'curtail%':>10} {'share%':>8}")
    print(f"  {'-'*4} {'-'*6} {'-'*11} {'-'*11} {'-'*10} {'-'*8}")

    for i in range(n_batteries):
        req  = result.per_battery_requested_kwh[i]
        curt = result.per_battery_curtailment_kwh[i]
        rate  = (curt / req  * 100) if req  > 0 else 0.0
        share = (curt / total_curtailed * 100) if total_curtailed > 0 else 0.0

        tag = " ← end of line" if positions[i] == 1.0 else (
              " ← near source" if positions[i] == 0.1 else "")

        print(f"  {i:<4} {positions[i]:>6.1f} {req:>11.2f} "
              f"{curt:>11.2f} {rate:>9.1f}% {share:>7.1f}%{tag}")

    print(f"\n  Peak: {compute_peak_kw(result.aggregate_load_kw):.1f} kW  |  "
          f"Violations: {count_feeder_violations(result.feeder_violations)}  |  "
          f"Excess: {compute_total_feeder_excess_kw(result.aggregate_load_kw, limit):.1f} kWh")

    for i in range(n_batteries):
        req  = result.per_battery_requested_kwh[i]
        curt = result.per_battery_curtailment_kwh[i]
        all_results.append({
            "scenario":          label,
            "feeder_limit_kw":   limit,
            "battery_id":        i,
            "position":          positions[i],
            "requested_kwh":     round(req,  3),
            "curtailed_kwh":     round(curt, 3),
            "curtail_rate_pct":  round((curt / req  * 100) if req  > 0 else 0.0, 2),
            "curtail_share_pct": round((curt / total_curtailed * 100)
                                       if total_curtailed > 0 else 0.0, 2),
        })

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df = pd.DataFrame(all_results)
output_dir  = Path("results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "fairness_sweep_results.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")