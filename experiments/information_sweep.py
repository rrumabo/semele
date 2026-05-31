import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.battery import Battery
from src.controllers import (
    local_aware_controller,
    neighborhood_controller,
    soft_capped_tou_controller,
    tou_controller,
)
from src.simulator import (
    compute_max_ramp_kw,
    compute_peak_kw,
    compute_total_feeder_excess_kw,
    count_feeder_violations,
    run_simulation,
)

np.random.seed(42)
random.seed(42)

HETEROGENEOUS_SOC   = True   # False = uniform 0.5, True = random 0.2–0.8
STOCHASTIC_PRICES   = False   # False = linspace, True = random uniform

# ---------------------------------------------------------------------------
# Configuration — same scenarios as Phase 1 for direct comparability
# ---------------------------------------------------------------------------

n_steps = 24

scenario_specs = [
    {"label": "light",  "n_batteries": 5,  "feeder_limit_kw": 60.0},
    {"label": "medium", "n_batteries": 10, "feeder_limit_kw": 60.0},
    {"label": "hard",   "n_batteries": 20, "feeder_limit_kw": 80.0},
]

# Four information levels, each mapped to a controller.
# The controller determines what the agent actually uses — the simulator
# always passes all available context.
information_levels: dict[str, object] = {
    "none":         tou_controller,
    "local":        local_aware_controller,
    "neighborhood": neighborhood_controller,
    "global":       soft_capped_tou_controller,   # uses feeder_load_kw + feeder_limit_kw
}

# One shared synthetic dataset. Each scenario slices the columns it needs.
loads_master = np.random.uniform(
    5, 15, size=(n_steps, max(s["n_batteries"] for s in scenario_specs))
)
prices = (np.random.uniform(10, 100, n_steps) if STOCHASTIC_PRICES
          else np.linspace(10, 100, n_steps))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batteries(n: int) -> list[Battery]:
    return [
        Battery(capacity_kwh=50, max_charge_kw=10, max_discharge_kw=10,
                initial_soc=(random.uniform(0.2, 0.8) if HETEROGENEOUS_SOC else 0.5))
        for _ in range(n)
    ]

def run_one(
    level_name: str,
    controller,
    loads: np.ndarray,
    n_batteries: int,
    feeder_limit_kw: float,
    scenario_label: str,
) -> dict:
    result = run_simulation(
        load_profiles_kw = loads,
        prices           = prices,
        batteries        = make_batteries(n_batteries),
        controller       = controller,
        dt_hours         = 1.0,
        low_threshold    = 30,
        high_threshold   = 70,
        feeder_limit_kw  = feeder_limit_kw,
    )
    return {
        "scenario":          scenario_label,
        "n_batteries":       n_batteries,
        "feeder_limit_kw":   feeder_limit_kw,
        "information_level": level_name,
        "peak":              compute_peak_kw(result.aggregate_load_kw),
        "ramp":              compute_max_ramp_kw(result.aggregate_load_kw),
        "violations":        count_feeder_violations(result.feeder_violations),
        "total_excess":      compute_total_feeder_excess_kw(result.aggregate_load_kw, feeder_limit_kw),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

all_results: list[dict] = []

for spec in scenario_specs:
    label = spec["label"]
    n     = spec["n_batteries"]
    limit = spec["feeder_limit_kw"]
    loads = loads_master[:, :n]

    print(f"\n{'='*62}")
    print(f"Scenario: {label:<8} | n_batteries={n:>2}  feeder_limit={limit:.0f} kW")
    print(f"{'='*62}")
    print(f"  {'level':<14} {'peak':>7}  {'ramp':>6}  {'violations':>10}  {'excess':>8}")
    print(f"  {'-'*14} {'-'*7}  {'-'*6}  {'-'*10}  {'-'*8}")

    for level_name, ctrl in information_levels.items():
        row = run_one(level_name, ctrl, loads, n, limit, label)
        all_results.append(row)
        print(
            f"  {level_name:<14} "
            f"{row['peak']:7.1f}  "
            f"{row['ramp']:6.1f}  "
            f"{row['violations']:>10d}  "
            f"{row['total_excess']:8.1f}"
        )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df = pd.DataFrame(all_results)[[
    "scenario", "n_batteries", "feeder_limit_kw",
    "information_level", "peak", "ramp", "violations", "total_excess",
]]

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
soc_tag   = "hetero_soc"   if HETEROGENEOUS_SOC else "uniform_soc"
price_tag = "stoch_prices" if STOCHASTIC_PRICES  else "det_prices"
output_path = output_dir / f"information_sweep_{soc_tag}_{price_tag}.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved: {output_path}")