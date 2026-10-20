# Small edits to the existing experiment scripts

These are deletions only. No logic changed.

| File | Edit |
|---|---|
| `reproduce_core_result.py` | Removed `NETWORK_ALPHA`, `NETWORK_K`, and their entries in the JSON summary. Removed `k=` and `alpha=` from the three topology builders. Removed `generation_kw=pv_kw`. `make_price_signal` now calls `pricing.pv_shaped_price`, which is the same formula. |
| `run_null_topology_test.py` | Dropped the `NETWORK_K` and `NETWORK_ALPHA` imports. Changed `make_linear(N, k=…, alpha=…)` to `make_linear(N)`. Removed `generation_kw`. |
| `make_paper_plots.py`, `check_consensus_mode.py`, `run_belief_rho_multiseed.py`, `run_belief_rho_sweep.py`, `run_spectral_sweep.py` | Removed `generation_kw=…`. |

**Why:** those arguments only fed the parts that were removed (the frequency model and the old agent dynamics).

**Check:** all of these scripts were run before and after the edits. Every printed line is identical, and every output CSV agrees to within 1e-13.

**Note:** these scripts must be run as modules from the repo root, for example `python -m experiments.make_paper_plots`. They import from `experiments.…`. That was already true before.

**`experiments/legacy/`:** no longer runs, because it used the removed controllers and the frequency model. It's in `.gitignore`, so it was never published. It's still in your git history. You can delete the folder.
