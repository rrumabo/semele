# network.py — block by block

**What it does:** stores who sees whom, which is the information graph between batteries.

---

## What was removed, and why
- The `Agent` dynamics: `state`, `step`, `tanh` power, `alpha`, `k`, `p_max`, `capacity`, and `soc`.
- The `Edge` class, `Network.step`, `neighbor_states`, `state_vector`, and `power_vector`.
- The simulator only ever read `agents[i].neighbors`. Everything else was an older consensus model that no current experiment runs.
- So `make_linear`, `make_star` and `make_small_world` no longer take `k` and `alpha`. The experiment scripts were updated to match.
- **Check:** the graphs come out identical, because the rewiring draws the random numbers in the same order. The paper numbers are unchanged.

## `Agent`
- An id and a list of neighbour ids. Nothing else.

## `Network`
- A dictionary from id to agent.
- **`add_edge(a, b)`** adds b to a's list and a to b's list, skipping duplicates. "Undirected" means if a sees b, then b sees a.
- **`laplacian()`** (new) builds L = D − A, the matrix in the paper's Lemma 1.
  - A[i, j] = 1 if i and j are neighbours.
  - D is the diagonal of neighbour counts.
  - It isn't used yet. It's there so later spectral checks can use the exact graph the simulator ran on.

## `_empty(n)`
- Makes n agents with no links. The three builders below start from it.

## `make_linear`
- A chain: 0–1, 1–2, and so on.

## `make_star`
- 0 is the hub, and everyone else links only to 0.

## `make_small_world`
Watts–Strogatz.
1. Start from a ring where each node links to its `k_nn` nearest neighbours (half on each side).
2. For each ring link, with probability `p`, throw it away and link to a random node instead.
   - It retries to avoid self-links and duplicates.
   - `seed` fixes the randomness, so the same graph comes out every time.
