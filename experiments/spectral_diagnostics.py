"""
Spectral (SVD) dimensionality diagnostic for Semele.

Runs directly on SimulationResult.per_battery_power_kw, which has shape (T, N):
    axis 0 = timestep
    axis 1 = battery / agent

Purpose
-------
Measure how many *active collective dimensions* the fleet occupies as rho grows.

    w1 ~ 1   -> one dominant mode: the common (broadcast) channel.
                The mean field is a sufficient description. MFG regime holds.
    w1 falls -> energy spreads across modes: topology has re-activated.
                The mean no longer suffices.

The knee of w1(rho) is rho_dim: where the 1-D description breaks.
Compare it against rho_c (collapse). If rho_dim < rho_c, there is a
warning zone [rho_dim, rho_c] where structure has emerged but the system
has not yet collapsed -- an early-warning signal the mean field cannot see.
"""

from __future__ import annotations
import numpy as np


def spectral_diagnostics(power_TN: np.ndarray, subtract_mean: bool = True):
    """
    Parameters
    ----------
    power_TN : np.ndarray, shape (T, N)
        Per-battery power trajectory for ONE rho value.
        This is exactly SimulationResult.per_battery_power_kw.
    subtract_mean : bool
        If True (default), remove the common field at each timestep so the
        SVD sees only each agent's DEVIATION from the broadcast signal.
        This is what isolates topology from the common channel.

    Returns
    -------
    w1 : float   fraction of energy in the dominant singular mode
    PR : float   participation ratio (effective number of active modes)
    u1 : np.ndarray, shape (N,)
         dominant spatial mode: how the leading mode distributes over agents.
         near-uniform  -> everyone moves together (broadcast)
         structured    -> groups of agents with opposite sign (topology returning)
    """
    X = np.asarray(power_TN, dtype=float)          # (T, N)

    if subtract_mean:
        # remove the common field: mean across agents, per timestep
        X = X - X.mean(axis=1, keepdims=True)

    # SVD. Columns of Vt correspond to agent-space modes.
    # We want the agent-space singular vectors, so operate on X (T,N):
    #   X = U (T,r) . S (r) . Vt (r,N)
    # Vt[0] is the dominant AGENT-space mode -> that's our u1.
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    energy = s ** 2
    total = energy.sum()

    if total <= 0.0:
        # X is all zeros after mean subtraction => perfect synchrony,
        # i.e. one mode (the common one we just removed). Treat as 1-D.
        n = X.shape[1]
        return 1.0, 1.0, np.ones(n) / np.sqrt(n)

    w1 = float(energy[0] / total)
    PR = float((total ** 2) / (energy ** 2).sum())
    u1 = np.asarray(Vt[0])                          # (N,)

    return w1, PR, u1


def sweep_over_rho(trajectories: dict[float, np.ndarray], subtract_mean: bool = True):
    """
    Parameters
    ----------
    trajectories : dict {rho_value: power_TN}
        One (T, N) array per rho. Use ONE representative seed per rho --
        the diagnostic needs a trajectory, not statistics.

    Returns
    -------
    rhos : np.ndarray
    w1s  : np.ndarray
    PRs  : np.ndarray
    u1s  : list[np.ndarray]   dominant spatial mode per rho
    """
    rhos = np.array(sorted(trajectories))
    w1s, PRs, u1s = [], [], []
    for r in rhos:
        w1, PR, u1 = spectral_diagnostics(trajectories[r], subtract_mean=subtract_mean)
        w1s.append(w1)
        PRs.append(PR)
        u1s.append(u1)
        print(f"rho={r:6.3f}   w1={w1:6.3f}   PR={PR:6.2f}")
    return rhos, np.array(w1s), np.array(PRs), u1s


def find_knee(rhos: np.ndarray, w1s: np.ndarray, drop: float = 0.05) -> float | None:
    """
    Crude knee finder: first rho where w1 has fallen by `drop` below its
    low-rho plateau (mean of first few points). Returns rho_dim, or None
    if w1 never drops (broadcast dominates the whole sweep -> hard crossover).
    """
    if len(rhos) < 3:
        return None
    plateau = w1s[:max(2, len(w1s) // 10)].mean()
    below = np.where(w1s < plateau - drop)[0]
    return float(rhos[below[0]]) if len(below) else None


if __name__ == "__main__":
    # ---- self-test on synthetic data so you can see expected behaviour ----
    # Build fake (T,N) trajectories where rho controls how much a common
    # shock dominates over private per-agent motion.
    rng = np.random.default_rng(0)
    T, N = 200, 30
    fake = {}
    for rho in np.linspace(0.0, 1.0, 11):
        common = rng.normal(0, 1, (T, 1))                 # shared mode
        private = rng.normal(0, 1, (T, N))                # independent
        X = np.sqrt(rho) * common + np.sqrt(1 - rho) * private
        fake[round(float(rho), 3)] = X

    print("Synthetic sanity check (w1 should RISE toward 1 as rho -> 1,")
    print("because a shared shock = one dominant mode):\n")
    rhos, w1s, PRs, u1s = sweep_over_rho(fake, subtract_mean=False)
    print("\nNote: with subtract_mean=False the common mode is visible and")
    print("w1 climbs to ~1 at rho=1. With subtract_mean=True you instead")
    print("watch the RESIDUAL structure after removing the broadcast.")