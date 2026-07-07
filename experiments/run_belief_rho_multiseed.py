"""
Multi-seed 2D sweep: belief_weight x rho, with bootstrap CI on topology spread.

Why
---
The single-seed sweep suggested topology spread grows as belief_weight falls
(0.022 at bw=0.5  ->  0.075 at bw=0.0). But single-seed spreads of ~0.02 could
be pure noise. This script runs MANY seeds per (belief_weight, rho) cell,
averages w1 across seeds, and puts a bootstrap confidence interval on the
resulting topology spread so you know which differences survive.

Only the noise (Z, eps drawn inside run_simulation) varies across seeds.
The world (PV, prices, load) and the graph structure are held fixed, so the
seed isolates forecast-noise realisation -- exactly the quantity rho controls.

Key quantities per (belief_weight, rho):
    w1_mean[topo]  -- mean dominant-mode fraction over seeds
    spread          = max_topo(w1_mean) - min_topo(w1_mean)
    spread_ci       -- bootstrap 95% CI on that spread

Reported per belief_weight:
    max spread over rho, with CI, and whether its lower bound clears a
    noise floor (so you can say "real" vs "could be noise").

Run from repo root (after `pip install -e .`):
    python -m experiments.run_belief_rho_multiseed
"""

from __future__ import annotations
import functools
import numpy as np

from experiments.reproduce_core_result import (
    make_cyprus_style_pv_profile,
    make_price_signal,
    make_load_profiles,
    make_batteries,
    build_topologies,
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
)
from semele.controllers import belief_neighborhood_controller
from semele.simulator import run_simulation

from experiments.spectral_diagnostics import spectral_diagnostics


SUBTRACT_MEAN = False
N_SEEDS = 30                       # seeds per (belief_weight, rho, topology) cell
BASE_SEED = 1000                   # seeds will be BASE_SEED + 0 .. N_SEEDS-1
N_BOOTSTRAP = 2000
BELIEF_WEIGHTS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.0]

# Fixed world, built once (identical across everything). Only noise varies.
_PV = make_cyprus_style_pv_profile(n_steps=N_STEPS, pv_peak_kw=PV_PEAK_KW)
_PRICES = make_price_signal(_PV)
_LOAD = make_load_profiles(
    rng=np.random.default_rng(42), n_steps=N_STEPS, n_batteries=N_BATTERIES,
)


def w1_for_cell(topology_factory, belief_weight: float, rho: float, noise_seed: int) -> float:
    """One w1 value for a single (topology, belief_weight, rho, seed)."""
    controller = functools.partial(
        belief_neighborhood_controller, belief_weight=belief_weight,
    )
    batteries = make_batteries(
        rng=np.random.default_rng(42), n_batteries=N_BATTERIES,
    )
    network = topology_factory()

    np.random.seed(noise_seed)     # ONLY the forecast noise changes with seed

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


def bootstrap_spread_ci(w1_by_topo_seed: dict[str, np.ndarray], n_boot: int, rng):
    """
    w1_by_topo_seed: {topology: array of w1 over seeds} for ONE (bw, rho) cell.
    Returns (spread_point, lo, hi) where spread = max-min of per-topo means,
    with a bootstrap 95% CI over seeds.
    """
    topos = list(w1_by_topo_seed)
    stacked = np.vstack([w1_by_topo_seed[t] for t in topos])   # (n_topo, n_seeds)
    n_seeds = stacked.shape[1]

    point = stacked.mean(axis=1)
    spread_point = float(point.max() - point.min())

    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, n_seeds)     # resample seeds with replacement
        m = stacked[:, idx].mean(axis=1)
        boots[b] = m.max() - m.min()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return spread_point, float(lo), float(hi)


def main() -> None:
    topologies = build_topologies()
    rng = np.random.default_rng(7)
    seeds = [BASE_SEED + i for i in range(N_SEEDS)]

    print(f"Multi-seed 2D sweep  (N_SEEDS={N_SEEDS}, bootstrap={N_BOOTSTRAP}, "
          f"subtract_mean={SUBTRACT_MEAN})")
    print("spread = max-min of per-topology mean w1; [lo, hi] = bootstrap 95% CI\n")

    for bw in BELIEF_WEIGHTS:
        best = (-1.0, 0.0, 0.0, None)   # spread, lo, hi, rho
        for rho in RHO_VALUES:
            w1_by_topo = {}
            for name, factory in topologies.items():
                w1_by_topo[name] = np.array(
                    [w1_for_cell(factory, bw, rho, s) for s in seeds]
                )
            sp, lo, hi = bootstrap_spread_ci(w1_by_topo, N_BOOTSTRAP, rng)
            if sp > best[0]:
                best = (sp, lo, hi, float(rho))

        sp, lo, hi, rho_star = best
        verdict = "REAL (CI clears 0.01)" if lo > 0.01 else "could be noise"
        print(f"belief_weight={bw:4.2f}   max spread={sp:.4f}  "
              f"CI=[{lo:.4f}, {hi:.4f}]  at rho={rho_star:.2f}   -> {verdict}")


if __name__ == "__main__":
    main()