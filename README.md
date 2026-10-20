# Lampyris

A small simulator of home batteries that all respond to the same noisy price signal.

The question:

> When every battery in a street reacts to the same forecast, and the forecast errors are correlated, what does the fleet do together — and does the communication network between them still matter?

The repository is called **Lampyris**. The Python package inside it is called **`semele`**.

---

## 1. Published result

**Paper:** S. Savva, *When a common price signal is present, network topology leaves no fingerprint on a storage fleet's collective dynamics*, arXiv:2607.06381 (nlin.CD), July 2026.

**Setup**

Each battery sees the price through its own forecast error:

```
eps_i(t) = sqrt(rho) * Z(t) + sqrt(1 - rho) * eta_i(t)
```

- `Z` is shared by everyone.
- `eta_i` is private to each battery.
- `rho` is the correlation between any two batteries' errors.

**Key fact**

The mean error does not average away:

```
Var(mean error) = (sigma^2 / N) * [1 + (N - 1) * rho]  ->  rho * sigma^2   as N -> infinity
```

**Lemma (linearised fleet)**

The shared part of the error drives only the consensus mode `1/sqrt(N)` of the graph Laplacian. No wiring can change that mode, so the topology's share of the dynamics is O(1/(rho N)).

**Measured** (20 batteries; linear, star and small-world graphs)

| | rho = 0 | rho = 1 |
|---|---|---|
| `w1` (energy in the dominant mode) | 0.43–0.44 | 0.963 |
| `PR` (effective number of modes) | 4.3–4.4 | 1.08 |
| alignment of the dominant mode with `1/sqrt(N)` | 0.96 | 0.998 |

- The three topologies give one curve. The change from rho = 0 to rho = 1 is a smooth crossover, not a phase transition.
- Between-topology spread is 0.0043 (alpha = 0.5) and 0.0181 (alpha = 0). Both are below the same-graph noise floor of about 0.03, measured over 30 seeds with independent noise per replica.

**What did not survive**

An earlier version of this project claimed a *collapse threshold* near rho ≈ 0.84. The core sweep falsified it. Feeder violations rise smoothly with rho, with no knee and no bimodality. Correlation is a continuous risk dial, not a cliff.

## 2. Current work: a PV energy community (on trial)

`experiments/run_community.py` puts the same fleet in a street with rooftop PV:

- **PV counted in the feeder.** The flow is `load − PV − battery`. Charging during a solar surplus relieves the feeder instead of loading it. Import and export limits are both checked.
- **Two price rules:**
  - `fixed`: the paper's rule, set in advance from the PV shape.
  - `sdr`: a local price set every hour by the street's own supply and demand (the supply–demand-ratio mechanism of Liu et al., 2017), using the batteries' actions from the previous hour.
- **Realistic rho.** Houses under the same clouds make rho high, so the sweep covers 0.7–1, plus 0 as a reference.
- **Battery size** is swept at 3, 5, 7 and 10 kW per house against a 6 kW per-house feeder limit.

**First results (synthetic data, 100 runs per condition)**

- **Fixed price.** 3–5 kW batteries cause almost no violations. At 7–10 kW, violations appear from rho ≈ 0.7 and rise smoothly. At 10 kW and rho = 1, the batteries cause violations in 19% of hours.
- **SDR price.**
  - The peak is unchanged, but the price flip-flops 80 → 20 → 80 from hour to hour, even at rho = 0.
  - The mechanism: discharging counts as supply, the price drops, everyone charges, the price rises, and the cycle repeats.
  - The oscillation grows with battery size, not with rho. At 5 kW, the battery-caused violations rise from 0.7% to 18.5%.
  - **Open:** this rests on two modelling choices, a one-hour price lag and an all-or-nothing controller. Whether it survives a same-hour clearing or a proportional controller is the next test.

## 3. Repository

```
src/semele/
  battery.py       battery with power, energy and efficiency limits
  controllers.py   TOU controller + the paper's belief/neighbour hybrid
  network.py       information graph (linear, star, small world) + Laplacian
  pricing.py       fixed PV-shaped price and SDR local price
  simulator.py     one day of the fleet behind one feeder

experiments/
  reproduce_core_result.py     rho sweep: peaks, violations, curtailment
  make_paper_plots.py          paper figures 1 and 2
  check_consensus_mode.py      dominant mode vs 1/sqrt(N)
  run_null_topology_test.py    same-graph noise floor
  run_belief_rho_sweep.py      2D sweep over rho and alpha
  run_belief_rho_multiseed.py  the same, over 30 seeds
  run_spectral_sweep.py        w1 / PR along rho
  spectral_diagnostics.py      SVD measures w1 and PR
  run_community.py             PV community: price rules x battery size x rho

explanations/      a plain-language walkthrough of every code file
```

## 4. Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m experiments.make_paper_plots      # results/paper/  (png + csv)
python -m experiments.run_community         # results/community/  (about 1 min)
```

Run scripts as modules (`python -m experiments.…`), because they import from each other. Outputs go to `results/`, which is not tracked by git.

## 5. Limits

- **Synthetic profiles.** These are not real Cyprus data: one day of PV and load, 20 houses, hourly steps.
- **No AC power flow.** The feeder is a single import/export limit.
- **Rule-based controllers.** These are not optimising controllers.
- **The SDR mechanism** is implemented from the published formula, with a one-step lag. It is not tested against a real community's clearing.

## License

MIT. See `LICENSE`.
