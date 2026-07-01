"""
Phase 11 PROBE — minimal consensus dynamics, NO controller.

Implements the clean dynamics that already lives in network.py (Agent.step),
but with degree-normalized coupling so it is stable on every topology
(the as-written SUM form diverges on hubs):

    x_i(t+1) = (1-alpha) x_i
             + k_coupling * ( mean_{j~i} x_j  -  x_i )     # consensus pull
             + alpha * d_i(t)                              # forcing
    x_i clipped to [-1, 1]
    P_i = p_max * tanh(x_i)

Forcing is the SAME correlated process as the simulator:
    d_i(t) = sigma_f * ( sqrt(rho) Z(t) + sqrt(1-rho) eps_i(t) )

Two questions this answers, on the real repo:
  Q1  Does topology now move rho_c a LOT (vs the controller's ~2%)?
  Q2  Is the transition CRITICAL — does seed-variance of the order parameter
      peak near rho_c (true transition) or stay flat (smooth crossover)?
"""
from __future__ import annotations
import numpy as np
import networkx as nx

ALPHA = 0.30
K_COUPLING = 0.50      # stable for all graphs: needs k < (2-alpha)/2 = 0.85
P_MAX = 5.0
SIGMA_F = 1.0
N = 60
N_STEPS = 120
BURN_IN = 40           # discard transient before measuring stationary spread
N_SEEDS = 40
RHO_GRID = np.linspace(0.0, 1.0, 21)


def topologies(n=N):
    return {
        "ring":        nx.cycle_graph(n),
        "path":        nx.path_graph(n),
        "small_world": nx.watts_strogatz_graph(n, 4, 0.1, seed=1),
        "reg4":        nx.random_regular_graph(4, n, seed=1),
        "star":        nx.star_graph(n - 1),
    }


def graph_stats(G):
    G = nx.convert_node_labels_to_integers(G)
    k = np.array([d for _, d in G.degree()], float)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    ev = np.sort(np.linalg.eigvalsh(L))
    return {"mean_inv_deg": float(np.mean(1.0 / k)),
            "lambda_2": float(ev[1]),
            "mean_degree": float(k.mean())}


def row_stochastic(G):
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G)
    d = A.sum(axis=1)
    d[d == 0] = 1.0
    return A / d[:, None]                    # P = D^{-1} A  (mean over neighbours)


def run_one(P, rho, seed):
    """One realization; returns sync_metric = mean_t std_i(power) over stationary part."""
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.1, 0.1, size=n)       # small random IC
    powers = np.empty((N_STEPS, n))
    Z = rng.standard_normal(N_STEPS)
    eps = rng.standard_normal((N_STEPS, n))
    noise = np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * eps
    d = SIGMA_F * noise
    for t in range(N_STEPS):
        consensus = P @ x - x                # mean neighbour - self
        x = (1 - ALPHA) * x + K_COUPLING * consensus + ALPHA * d[t]
        x = np.clip(x, -1.0, 1.0)
        powers[t] = P_MAX * np.tanh(x)
    stat = powers[BURN_IN:]
    sync = float(np.mean(np.std(stat, axis=1)))
    # coincidence / extreme observable: worst-case aggregate pull (feeder stress)
    agg = stat.mean(axis=1)                       # mean power across agents per step
    peak_coincidence = float(np.max(np.abs(agg)) / P_MAX)   # in [0,1]
    return sync, peak_coincidence


def midpoint(rho_grid, sync_mean):
    """Quick rho_c: where collapse_score crosses 0.5, linearly interpolated."""
    lo, hi = sync_mean[:2].mean(), sync_mean[-2:].mean()
    if abs(lo - hi) < 1e-9:
        return np.nan
    collapse = 1 - (sync_mean - hi) / (lo - hi)   # 0 at low rho, 1 at high rho
    for i in range(len(rho_grid) - 1):
        if collapse[i] <= 0.5 <= collapse[i + 1]:
            f = (0.5 - collapse[i]) / (collapse[i + 1] - collapse[i] + 1e-12)
            return float(rho_grid[i] + f * (rho_grid[i + 1] - rho_grid[i]))
    return float(rho_grid[np.argmin(np.abs(collapse - 0.5))])


def main():
    tops = topologies()
    print(f"N={N}  alpha={ALPHA}  k_coupling={K_COUPLING}  seeds={N_SEEDS}  steps={N_STEPS}\n")
    print(f"{'topology':12s} {'mean_inv_deg':>12s} {'lambda_2':>9s} {'rho_c':>7s} "
          f"{'sync@0':>7s} {'sync@1':>7s} {'var_peak_rho':>13s} {'peak/floor':>11s}")

    summary = {}
    for name, G in tops.items():
        gs = graph_stats(G)
        Pmat = row_stochastic(G)
        sync_curve, coin_curve = [], []
        for rho in RHO_GRID:
            pairs = np.array([run_one(Pmat, rho, 7000 + s) for s in range(N_SEEDS)])
            sync_curve.append(pairs[:, 0].mean())
            coin_curve.append(pairs[:, 1].mean())
        sync_curve = np.array(sync_curve); coin_curve = np.array(coin_curve)
        summary[name] = dict(gs=gs, sync=sync_curve, coin=coin_curve)
        print(f"{name:12s} mid={gs['mean_inv_deg']:.3f} l2={gs['lambda_2']:6.3f} | "
              f"sync@.5={sync_curve[10]:.3f}  coincidence@.5={coin_curve[10]:.3f}  "
              f"coincidence@0={coin_curve[0]:.3f}")

    def spread(key):
        vals = [s[key][10] for s in summary.values()]   # at rho=0.5
        return max(vals) - min(vals), vals
    s_sync, _ = spread("sync")
    s_coin, coin_vals = spread("coin")
    print(f"\n@ rho=0.5 across topologies:")
    print(f"  sync_metric spread        = {s_sync:.3f}")
    print(f"  peak_coincidence spread   = {s_coin:.3f}   values={[round(v,3) for v in coin_vals]}")
    print(f"  -> structure lives in the {'COINCIDENCE/tail' if s_coin > 2*s_sync else 'neither (mean-field)'} observable")
    return summary


def _old_main_unused():
    tops = topologies()


if __name__ == "__main__":
    main()