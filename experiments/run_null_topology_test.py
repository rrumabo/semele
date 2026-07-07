"""
Null-topology test (CORRECTED): same graph, DIFFERENT noise seed per replica.

What was wrong before
---------------------
The first null gave the three identical 'linear' graphs the SAME noise seed
per seed-index. Same graph + same seed = identical trajectories = spread 0 by
construction. That measured determinism, not the noise floor.

Correct design
--------------
The floor we want is: "with topology held FIXED, how much does w1 move when
only the forecast-noise realisation changes?" So each of the three replicas is
the same linear graph but is fed an INDEPENDENT noise seed. The spread between
them is then pure noise-realisation variability -- the true floor a real
topology effect must beat.

Concretely, for seed-index i we give:
    replica A -> seed  BASE + 3*i + 0
    replica B -> seed  BASE + 3*i + 1
    replica C -> seed  BASE + 3*i + 2
so all three see different noise, same graph.

Compare the resulting null-spread to the real between-topology spreads
(0.0043 at bw=0.5, 0.0181 at bw=0.0). A real spread is meaningful only if it
exceeds this noise floor.

Run from repo root:
    python -m experiments.run_null_topology_test_v2
"""

from __future__ import annotations
import functools
import numpy as np

from experiments.reproduce_core_result import (
    make_cyprus_style_pv_profile,
    make_price_signal,
    make_load_profiles,
    make_batteries,
    N_STEPS,
    N_BATTERIES,
    RHO_VALUES,
    DT_HOURS,
    PV_PEAK_KW,
    FEEDER_LIMIT_KW,
    LOW_PRICE,
    HIGH_PRICE,
    FORECAST_ERROR_SIGMA_KW,
    PRICE_SIGMA,
    NETWORK_K,
    NETWORK_ALPHA,
)
from semele.network import make_linear
from semele.controllers import belief_neighborhood_controller
from semele.simulator import run_simulation

from experiments.spectral_diagnostics import spectral_diagnostics


SUBTRACT_MEAN = False
N_SEEDS = 30
BASE_SEED = 1000
N_BOOTSTRAP = 2000
BELIEF_WEIGHTS = [0.5, 0.0]

_PV = make_cyprus_style_pv_profile(n_steps=N_STEPS, pv_peak_kw=PV_PEAK_KW)
_PRICES = make_price_signal(_PV)
_LOAD = make_load_profiles(
    rng=np.random.default_rng(42), n_steps=N_STEPS, n_batteries=N_BATTERIES,
)

# One fixed graph. All three replicas use THIS SAME structure.
def _linear():
    return make_linear(N_BATTERIES, k=NETWORK_K, alpha=NETWORK_ALPHA)

REPLICAS = ["A", "B", "C"]   # same graph, different noise seeds


def w1_for_cell(belief_weight, rho, noise_seed):
    controller = functools.partial(
        belief_neighborhood_controller, belief_weight=belief_weight,
    )
    batteries = make_batteries(rng=np.random.default_rng(42), n_batteries=N_BATTERIES)
    network = _linear()
    np.random.seed(noise_seed)
    result = run_simulation(
        load_profiles_kw=_LOAD,
        prices=_PRICES,
        batteries=batteries,
        controller=controller,
        dt_hours=DT_HOURS,
        low_threshold=LOW_PRICE,
        high_threshold=HIGH_PRICE,
        feeder_limit_kw=FEEDER_LIMIT_KW,
        rho_agents=float(rho),
        forecast_error_sigma_kw=FORECAST_ERROR_SIGMA_KW,
        price_sigma=PRICE_SIGMA,
        network=network,
        generation_kw=_PV,
    )
    w1, _, _ = spectral_diagnostics(result.per_battery_power_kw, subtract_mean=SUBTRACT_MEAN)
    return w1


def bootstrap_spread_ci(w1_by_replica, n_boot, rng):
    labels = list(w1_by_replica)
    stacked = np.vstack([w1_by_replica[t] for t in labels])
    n_seeds = stacked.shape[1]
    point = stacked.mean(axis=1)
    spread_point = float(point.max() - point.min())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, n_seeds)
        m = stacked[:, idx].mean(axis=1)
        boots[b] = m.max() - m.min()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return spread_point, float(lo), float(hi)


def max_null_spread(belief_weight, rng):
    best = (-1.0, 0.0, 0.0, None)
    for rho in RHO_VALUES:
        w1_by = {r: np.empty(N_SEEDS) for r in REPLICAS}
        for i in range(N_SEEDS):
            for j, r in enumerate(REPLICAS):
                # same graph, independent noise seed per replica
                seed = BASE_SEED + 3 * i + j
                w1_by[r][i] = w1_for_cell(belief_weight, float(rho), seed)
        sp, lo, hi = bootstrap_spread_ci(w1_by, N_BOOTSTRAP, rng)
        if sp > best[0]:
            best = (sp, lo, hi, float(rho))
    return best


def main() -> None:
    rng = np.random.default_rng(7)

    print(f"Null-topology test v2  (N_SEEDS={N_SEEDS}, same graph, "
          f"independent noise seed per replica)")
    print("Spread here = pure noise-realisation variability at FIXED topology.\n")

    real = {0.5: 0.0043, 0.0: 0.0181}

    for bw in BELIEF_WEIGHTS:
        sp, lo, hi, rho_star = max_null_spread(bw, rng)
        r = real[bw]
        verdict = ("REAL: topology spread exceeds noise floor"
                   if r > hi else
                   "NOT distinguishable: real spread within noise floor")
        print(f"belief_weight={bw:4.2f}")
        print(f"   null max-spread = {sp:.4f}   CI=[{lo:.4f}, {hi:.4f}]  at rho={rho_star:.2f}")
        print(f"   real spread     = {r:.4f}")
        print(f"   -> {verdict}\n")


if __name__ == "__main__":
    main()
