import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.battery import Battery
from src.controllers import neighborhood_controller, tou_controller, soft_capped_tou_controller
from src.simulator import (
    compute_max_ramp_kw,
    compute_peak_kw,
    compute_total_feeder_excess_kw,
    count_feeder_violations,
    run_simulation,
)

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# Most realistic setup from Phase 2 (hetero SoC, stochastic prices).
# We fix scenarios and vary only neighborhood_size.
# none and global are included as fixed reference points.
# ---------------------------------------------------------------------------

n_steps = 24

scenario_specs = [
    {"label": "light",  "n_batteries": 5,  "feeder_limit_kw": 60.0},
    {"label": "medium", "n_batteries": 10, "feeder_limit_kw": 60.0},
    {"label": "hard",   "n_batteries": 20, "feeder_limit_kw": 80.0},
]

# Neighborhood sizes to test. "all" means every other agent = n_batteries - 1.
NEIGHBORHOOD_SIZES = [1, 2, 5, 10, "all"]

loads_master = np.random.uniform(
    5, 15, size=(n_steps, max(s["n_batteries"] for s in scenario_specs))
)
prices = np.random.uniform(10, 100, n_steps)


def make_batteries(n: int) -> list[Battery]:
    return [
        Battery(capacity_kwh=50, max_charge_kw=10, max_discharge_kw=10,
                initial_soc=random.uniform(0.2, 0.8))
        for _ in range(n)
    ]


def run_one(controller, n_batteries, feeder_limit_kw, loads, neighborhood_size) -> dict:
    result = run_simulation(
        load_profiles_kw  = loads,
        prices            = prices,
        batteries         = make_batteries(n_batteries),
        controller        = controller,
        dt_hours          = 1.0,
        low_threshold     = 30,
        high_threshold    = 70,
        feeder_limit_kw   = feeder_limit_kw,
        neighborhood_size = neighborhood_size,
    )
    return {
        "peak":       compute_peak_kw(result.aggregate_load_kw),
        "ramp":       compute_max_ramp_kw(result.aggregate_load_kw),
        "violations": count_feeder_violations(result.feeder_violations),
        "excess":     compute_total_feeder_excess_kw(result.aggregate_load_kw, feeder_limit_kw),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

all_results = []

for spec in scenario_specs:
    label = spec["label"]
    n     = spec["n_batteries"]
    limit = spec["feeder_limit_kw"]
    loads = loads_master[:, :n]

    # Fixed references — neighborhood_size doesn't affect these
    ref_none   = run_one(tou_controller,             n, limit, loads, 2)
    ref_global = run_one(soft_capped_tou_controller, n, limit, loads, 2)

    print(f"\n{'='*64}")
    print(f"Scenario: {label:<8} | n_batteries={n:>2}  feeder_limit={limit:.0f} kW")
    print(f"{'='*64}")
    print(f"  {'controller':<22} {'size':>5}  {'peak':>7}  {'peak % none':>11}  {'excess':>8}")
    print(f"  {'-'*22} {'-'*5}  {'-'*7}  {'-'*11}  {'-'*8}")
    print(f"  {'none':<22} {'—':>5}  {ref_none['peak']:7.1f}  {'(baseline)':>11}  {ref_none['excess']:8.1f}")
    print(f"  {'global':<22} {'—':>5}  {ref_global['peak']:7.1f}  {ref_global['peak']/ref_none['peak']*100:10.1f}%  {ref_global['excess']:8.1f}")

    for size_label in NEIGHBORHOOD_SIZES:
        # Clamp to n-1 for "all" and for sizes larger than fleet
        actual_size = (n - 1) if size_label == "all" else min(size_label, n - 1)

        m = run_one(neighborhood_controller, n, limit, loads, actual_size)
        pct = m['peak'] / ref_none['peak'] * 100

        print(f"  {'neighborhood':<22} {actual_size:>5}  {m['peak']:7.1f}  {pct:10.1f}%  {m['excess']:8.1f}")

        all_results.append({
            "scenario":          label,
            "n_batteries":       n,
            "feeder_limit_kw":   limit,
            "neighborhood_size": actual_size,
            "peak":              m["peak"],
            "peak_pct_of_none":  pct,
            "ramp":              m["ramp"],
            "violations":        m["violations"],
            "excess":            m["excess"],
            "none_peak":         ref_none["peak"],
            "global_peak":       ref_global["peak"],
        })

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df = pd.DataFrame(all_results)
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "neighborhood_size_sweep_results.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
