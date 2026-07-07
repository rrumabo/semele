"""
Run the SVD dimensionality diagnostic across rho, reusing the exact
setup from reproduce_core_result.py (same PV, prices, batteries, topologies).

Design choices baked in
-----------------------
* subtract_mean = False
      We watch the COMMON mode. w1 climbs toward 1 as the broadcast channel
      dominates; its knee marks rho_dim, where the 1-D (mean-field) description
      stops being enough.

* fixed seed BEFORE each run (SEED, same for every rho)
      run_simulation seeds np.random internally for the correlated noise, so we
      set the seed immediately before EACH call. Same seed for every rho means
      the ONLY thing changing between points is rho itself -- Z (common shock)
      and eps (private noise) are identical draws, only their mixing
      sqrt(rho)*Z + sqrt(1-rho)*eps changes. That gives a clean w1(rho), not a
      noisy one. One representative trajectory per rho; add multi-seed error
      bars later if wanted.

Run from the repo so that `import semele` and the experiment module resolve,
e.g. from repo root:  python run_spectral_sweep.py
(adjust EXPERIMENT_IMPORT below if your module path differs).
"""

from __future__ import annotations
import numpy as np

# Reuse the experiment's own builders so the physics matches Phase 7 exactly.
# Adjust this import to wherever reproduce_core_result lives in your repo.

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

from experiments.spectral_diagnostics import sweep_over_rho, find_knee


SEED = 42                    # same seed before every run
SUBTRACT_MEAN = False        # watch the common/broadcast mode


def collect_trajectories(topology_name: str, topology_factory):
    """
    One representative (T, N) power trajectory per rho, for a single topology.
    """
    # Fixed world: PV, prices, load, batteries built ONCE so they are identical
    # across rho. Only rho changes between runs.
    pv_kw = make_cyprus_style_pv_profile(n_steps=N_STEPS, pv_peak_kw=PV_PEAK_KW)
    prices = make_price_signal(pv_kw)

    build_rng = np.random.default_rng(SEED)
    load_profiles_kw = make_load_profiles(
        rng=build_rng, n_steps=N_STEPS, n_batteries=N_BATTERIES,
    )
    batteries_template = make_batteries(rng=build_rng, n_batteries=N_BATTERIES)

    trajectories: dict[float, np.ndarray] = {}
    for rho in RHO_VALUES:
        # Fresh batteries each run (apply_power mutates their internal energy).
        batteries = make_batteries(
            rng=np.random.default_rng(SEED), n_batteries=N_BATTERIES,
        )
        network = topology_factory()

        # Seed np.random right before the call: correlated noise is drawn inside.
        np.random.seed(SEED)

        result = run_simulation(
            load_profiles_kw=load_profiles_kw,
            prices=prices,
            batteries=batteries,
            controller=belief_neighborhood_controller,
            dt_hours=DT_HOURS,
            low_threshold=LOW_PRICE,
            high_threshold=HIGH_PRICE,
            feeder_limit_kw=FEEDER_LIMIT_KW,
            rho_agents=float(rho),
            forecast_error_sigma_kw=FORECAST_ERROR_SIGMA_KW,
            price_sigma=PRICE_SIGMA,
            network=network,
            generation_kw=pv_kw,
        )

        trajectories[float(rho)] = result.per_battery_power_kw   # (T, N)

    return trajectories


def main() -> None:
    topologies = build_topologies()

    for topology_name, topology_factory in topologies.items():
        print(f"\n=== topology: {topology_name} "
              f"(subtract_mean={SUBTRACT_MEAN}, seed={SEED}) ===")
        trajectories = collect_trajectories(topology_name, topology_factory)
        rhos, w1s, PRs, u1s = sweep_over_rho(
            trajectories, subtract_mean=SUBTRACT_MEAN,
        )
        rho_dim = find_knee(rhos, w1s)
        if rho_dim is None:
            print(f"  -> no knee: broadcast dominates whole sweep (hard crossover). "
                  f"Compare to rho_c ~ 0.84.")
        else:
            print(f"  -> rho_dim (knee) = {rho_dim:.3f}   vs   rho_c ~ 0.84")
            if rho_dim < 0.80:
                print(f"     early-warning zone [{rho_dim:.2f}, ~0.84]: "
                      f"structure emerges before collapse.")


if __name__ == "__main__":
    main()