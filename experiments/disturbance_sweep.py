import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.battery import Battery
from src.controllers import droop_controller
from src.simulator import (
    compute_frequency_nadir,
    compute_frequency_peak,
    compute_frequency_recovery_time,
    compute_frequency_std,
    compute_peak_kw,
    run_simulation,
)

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

n_steps     = 24
n_batteries = 10
M           = 10.0       # fixed at stability boundary
omega_nominal = 50.0
drop_steps  = 3          # generation ramps down over 3 timesteps

disturbance_sizes = [5.0, 10.0, 15.0, 20.0, 25.0]
droop_gains       = [0.5, 1.0, 2.0, 5.0, 10.0]

loads_master  = np.random.uniform(1, 5, size=(n_steps, n_batteries))
prices        = np.random.uniform(10, 100, n_steps)


def make_batteries(n: int) -> list[Battery]:
    return [
        Battery(capacity_kwh=50, max_charge_kw=10, max_discharge_kw=10,
                initial_soc=random.uniform(0.2, 0.8))
        for _ in range(n)
    ]


def make_generation(disturbance_size: float) -> np.ndarray:
    """
    Generation starts at 30 kW, drops by disturbance_size
    starting at a random timestep, ramping down over drop_steps.
    """
    gen = np.ones(n_steps) * 30.0
    drop_start = random.randint(6, n_steps - drop_steps - 1)
    for s in range(drop_steps):
        fraction = (s + 1) / drop_steps
        gen[drop_start + s] = 30.0 - disturbance_size * fraction
    gen[drop_start + drop_steps:] = 30.0 - disturbance_size
    return gen, drop_start


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

all_results = []

for disturbance_size in disturbance_sizes:
    generation_kw, drop_start = make_generation(disturbance_size)

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
            omega_nominal    = omega_nominal,
        )

        all_results.append({
            "disturbance_kw":             disturbance_size,
            "drop_start_timestep":        drop_start,
            "droop_gain":                 droop_gain,
            "frequency_nadir_hz":         compute_frequency_nadir(result.omega_hz),
            "frequency_peak_hz":          compute_frequency_peak(result.omega_hz),
            "frequency_recovery_time_hr": compute_frequency_recovery_time(result.omega_hz),
            "frequency_std_hz":           compute_frequency_std(result.omega_hz),
            "peak_kw":                    compute_peak_kw(result.aggregate_load_kw),
        })

        print(
            f"  disturbance={disturbance_size:5.1f} kW  "
            f"drop_at=t{drop_start:02d}  "
            f"droop={droop_gain:5.1f}  "
            f"nadir={result.omega_hz.min():.3f} Hz  "
            f"std={result.omega_hz.std():.3f}"
        )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

df = pd.DataFrame(all_results)
output_dir  = Path("results")
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "disturbance_sweep_results.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")