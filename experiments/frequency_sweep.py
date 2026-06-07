import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import random
from src.battery import Battery
from src.controllers import (
    droop_controller,
    soft_capped_tou_controller
)
from src.simulator import (
    compute_frequency_peak,
    compute_frequency_nadir,
    compute_frequency_recovery_time,
    compute_frequency_std,
    compute_peak_kw,
    compute_total_feeder_excess_kw,
    count_feeder_violations,
    run_simulation,
)

np.random.seed(42)
random.seed(42)

#--------------------------------------------------
# Configuration
# --------------------------------------------------

n_steps = 24
n_batteries = 10
droop_gains = [0.5, 1.0, 2.0, 5.0, 10.0]
inertia_values = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 20.0, 30.0, 50.0]

loads_master = np.random.uniform(1, 5, size=(n_steps, n_batteries))
prices = np.random.uniform(10, 100, n_steps)
generation_kw = 30.0

def make_batteries(n: int) -> list[Battery]:
    return [
        Battery(capacity_kwh=50, max_charge_kw=10, max_discharge_kw=10,
                initial_soc=random.uniform(0.2, 0.8))
        for _ in range(n)
    ]

# --------------------------------------------------
# Sweep
# --------------------------------------------------

all_results = []

for M in inertia_values:
    for droop_gain in droop_gains:

        controller = lambda **kw: droop_controller(**kw, droop_gain=droop_gain)

        result = run_simulation(
            load_profiles_kw = loads_master,
            prices           = prices,
            batteries        = make_batteries(n_batteries),
            controller       = controller,
            dt_hours         = 1.0,
            feeder_limit_kw  = 60.0,
            generation_kw    = generation_kw,
            M                = M,
            omega_nominal    = 50.0,
        )

        all_results.append({
            "inertia":                    M,
            "droop_gain":                 droop_gain,
            "frequency_nadir_hz":         compute_frequency_nadir(result.omega_hz),
            "frequency_peak_hz":          compute_frequency_peak(result.omega_hz),
            "frequency_recovery_time_hr": compute_frequency_recovery_time(result.omega_hz),
            "frequency_std_hz":           compute_frequency_std(result.omega_hz),
            "peak_kw":                    compute_peak_kw(result.aggregate_load_kw),
        })

#--------------------------------------------------
# Save Results
#--------------------------------------------------

df = pd.DataFrame(all_results)
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "frequency_sweep_results.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")