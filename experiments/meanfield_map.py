"""
Phase 11 — the map that survives.

Topology turned out flat (mean-field), so we drop it and map the order
parameter on the plane that DOES govern it: ρ (common-mode belief correlation)
vs k_coupling (the consensus pull that fights diversity). One fixed topology
(reg4); ring/path/small_world give the same picture, which is the whole point.

Output: results/phase11_meanfield_map.png
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = 0.30
P_MAX = 5.0
SIGMA_F = 1.0
N = 60
N_STEPS = 120
BURN = 40
SEEDS = 20
RHOS = np.linspace(0.0, 1.0, 21)
KS = np.linspace(0.0, 0.80, 17)        # stay below stability bound (2-alpha)/2 = 0.85


def row_stochastic(G):
    A = nx.to_numpy_array(nx.convert_node_labels_to_integers(G))
    d = A.sum(1); d[d == 0] = 1.0
    return A / d[:, None]


def sync_metric(P, rho, k, seed):
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.1, 0.1, n)
    Z = rng.standard_normal(N_STEPS)
    eps = rng.standard_normal((N_STEPS, n))
    d = SIGMA_F * (np.sqrt(rho) * Z[:, None] + np.sqrt(1 - rho) * eps)
    pw = np.empty((N_STEPS, n))
    for t in range(N_STEPS):
        x = (1 - ALPHA) * x + k * (P @ x - x) + ALPHA * d[t]
        x = np.clip(x, -1, 1)
        pw[t] = P_MAX * np.tanh(x)
    return float(np.mean(np.std(pw[BURN:], axis=1)))


def main(out_path="results/phase11_meanfield_map.png"):
    G = nx.random_regular_graph(4, N, seed=1)
    P = row_stochastic(G)
    M = np.zeros((len(KS), len(RHOS)))
    for i, k in enumerate(KS):
        for j, rho in enumerate(RHOS):
            M[i, j] = np.mean([sync_metric(P, rho, k, 5000 + s) for s in range(SEEDS)])

    half = 0.5 * (M.max() + M.min())   # boundary between diverse and synchronized
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(M, origin="lower", aspect="auto",
                   extent=[RHOS[0], RHOS[-1], KS[0], KS[-1]], cmap="viridis")
    cs = ax.contour(RHOS, KS, M, levels=[half], colors="white", linewidths=2)
    ax.clabel(cs, fmt="threshold", fontsize=9)
    ax.set_xlabel("ρ   (common-mode belief correlation)")
    ax.set_ylabel("k_coupling   (consensus pull)")
    ax.set_title("Synchronization phase map\nsync_metric = mean_t std_i(P)   "
                 "(reg4, N=60; ring/path/small_world identical)")
    cb = plt.colorbar(im)
    cb.set_label("sync_metric   (bright = diverse actions, dark = synchronized)")
    ax.text(0.05, 0.72, "DIVERSE", color="white", fontsize=12, weight="bold")
    ax.text(0.70, 0.10, "SYNCHRONIZED", color="white", fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    print(f"sync range: {M.min():.3f} (synced) .. {M.max():.3f} (diverse), boundary at {half:.3f}")
    return M


if __name__ == "__main__":
    main()