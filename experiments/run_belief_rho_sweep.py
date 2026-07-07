"""
2D dimensionality sweep: belief_weight  x  rho.

Motivation
----------
At belief_weight = 0.5 the broadcast (price/belief) channel dominates so
strongly that topology never re-appears: w1(rho) rises smoothly and all
topologies are near-identical (confirmed by the 1D sweep). To make the
topology channel visible, we LOWER belief_weight -- shifting trust from the
common feeder-belief signal toward the neighbour (topology) observation.

For each belief_weight we run the full rho sweep and record:
    w1(rho)  -- dominant-mode energy fraction
    PR(rho)  -- participation ratio (effective active dimensions)

What to look for
----------------
* High belief_weight (~0.5, 1.0): broadcast wins, w1 high everywhere,
  topologies collapse together. No knee.
* As belief_weight drops: the neighbour channel gains weight. Below some
  threshold the topologies should start to DIVERGE, and w1(rho) should
  develop a knee -- the rho where the common signal finally loses to
  structure.

The belief_weight where divergence first appears is the crossover you're after.

Injection detail
-----------------
run_simulation does NOT forward belief_weight to the controller, so the
controller always used its default (0.5). We bind belief_weight up front with
functools.partial; the simulator then calls the already-parameterised
controller. No change to simulator.py.

Run from repo root (after `pip install -e .`):
    python -m experiments.run_belief_rho_sweep
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


SEED = 42
SUBTRACT_MEAN = False

# belief_weight values to test. 1.0 = trust broadcast only, 0.0 = trust
# neighbours only. We include low values because that's where topology
# is expected to re-emerge.
BELIEF_WEIGHTS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.0]


def w1_curve(topology_factory, belief_weight: float):
    """
    Return (rhos, w1s, PRs) for one topology at one belief_weight.
    World (PV, prices, load, batteries) is held fixed; only rho varies.
    """
    pv_kw = make_cyprus_style_pv_profile(n_steps=N_STEPS, pv_peak_kw=PV_PEAK_KW)
    prices = make_price_signal(pv_kw)

    build_rng = np.random.default_rng(SEED)
    load_profiles_kw = make_load_profiles(
        rng=build_rng, n_steps=N_STEPS, n_batteries=N_BATTERIES,
    )

    # Bind belief_weight into the controller (simulator won't pass it).
    controller = functools.partial(
        belief_neighborhood_controller, belief_weight=belief_weight,
    )

    rhos, w1s, PRs = [], [], []
    for rho in RHO_VALUES:
        batteries = make_batteries(
            rng=np.random.default_rng(SEED), n_batteries=N_BATTERIES,
        )
        network = topology_factory()
        np.random.seed(SEED)

        result = run_simulation(
            load_profiles_kw=load_profiles_kw,
            prices=prices,
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
            generation_kw=pv_kw,
        )

        w1, PR, _ = spectral_diagnostics(
            result.per_battery_power_kw, subtract_mean=SUBTRACT_MEAN,
        )
        rhos.append(float(rho))
        w1s.append(w1)
        PRs.append(PR)

    return np.array(rhos), np.array(w1s), np.array(PRs)


def topology_spread(curves_by_topo):
    """
    Max spread in w1 across topologies, per rho. Large spread = topology
    matters (channels have diverged); near-zero = topology irrelevant.
    """
    stacked = np.vstack([w for (_, w, _) in curves_by_topo.values()])
    return stacked.max(axis=0) - stacked.min(axis=0)


def main() -> None:
    topologies = build_topologies()

    print(f"2D sweep: belief_weight x rho  (seed={SEED}, "
          f"subtract_mean={SUBTRACT_MEAN})\n")

    for bw in BELIEF_WEIGHTS:
        print(f"\n############  belief_weight = {bw:.2f}  ############")
        curves = {}
        for name, factory in topologies.items():
            rhos, w1s, PRs = w1_curve(factory, bw)
            curves[name] = (rhos, w1s, PRs)

        # Per-rho spread across topologies: the divergence signal.
        spread = topology_spread(curves)
        max_spread = float(spread.max())
        rho_at_max = float(rhos[int(spread.argmax())])

        # Print compact w1 table across topologies.
        header = "  rho    " + "".join(f"{n[:10]:>12s}" for n in curves) + "     spread"
        print(header)
        for i, r in enumerate(rhos):
            row = f"  {r:4.2f}   " + "".join(
                f"{curves[n][1][i]:12.3f}" for n in curves
            )
            row += f"   {spread[i]:8.4f}"
            print(row)

        print(f"  -> max topology spread = {max_spread:.4f} at rho={rho_at_max:.2f}")
        if max_spread < 0.02:
            print("     topologies still collapsed together (broadcast wins).")
        else:
            print("     TOPOLOGIES DIVERGING -- structure channel is active here.")


if __name__ == "__main__":
    main()