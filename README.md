# Semele

**Semele** is a small computational research project about a simple question:

> When a decentralized fleet of batteries is driven by the same noisy forecast signal, can the shared noise dominate the collective dynamics so strongly that the communication topology becomes hard to detect?

The short answer explored here is **yes**.

The project studies decentralized battery agents connected through different communication graphs — linear, star, and small-world — while they respond to a shared PV-derived price / forecast signal. The main observation is that the fleet’s collective trajectory can collapse onto a low-dimensional consensus-like mode. When that happens, changing the communication topology does not visibly change the effective dimensionality of the fleet response.

This does **not** mean topology has no effect. It means that, in this regime, topology-dependent modes are masked by a stronger common stochastic component.

## The idea

In many decentralized energy systems, agents are treated as independent local decision makers. But in practice they often depend on the same external information sources:

- the same PV forecast,
- the same price signal,
- the same market estimate,
- the same feeder-level prediction,
- or the same forecasting provider.

If the forecast errors are correlated, the agents are no longer independent in the way the model might assume. They can move together, not because they communicate perfectly, but because they are all being pushed by the same wrong signal.

Semele tests this mechanism in a battery-fleet simulator.

The central mechanism is:

```text
shared forecast error
        ↓
common forcing of all agents
        ↓
dominant consensus-mode motion
        ↓
low effective dimensionality
        ↓
weak observable topology effect
```

## Why “Semele”?

In Greek mythology, Semele is destroyed when she sees Zeus in his full divine form. The metaphor here is not that topology disappears because it is fake. It disappears from observation because a stronger force overwhelms it.

In this project:

```text
Semele    → topology-dependent structure
Zeus      → shared stochastic forcing
```

The topology is still there. It is just outpowered in the measured collective response.

## What is measured?

The simulator records the per-agent battery power trajectory and stacks it into a matrix:

```text
X ∈ R^(time × agents)
```

The project then studies the singular spectrum of this trajectory.

Two diagnostics are used:

- `w1`: the fraction of trajectory variance carried by the dominant mode.
- `PR`: the spectral participation ratio, interpreted as an effective dimensionality.

When `w1 → 1` and `PR → 1`, the fleet is effectively moving along one dominant collective direction.

That is the signature of common-mode domination.

## Core claim

The claim is narrow:

> Under shared stochastic forecast forcing, topology-dependent modes can become observationally weak because the dominant variance is injected into the graph-invariant consensus mode.

This is not a design claim saying “do not build communication networks.”

It is a detectability claim:

> If many agents share the same noisy information source, then changing the communication graph may matter less than reducing the correlation of the forecast errors.

## Repository structure

```text
src/semele/
  battery.py
  controllers.py
  network.py
  simulator.py

experiments/
  reproduce_core_result.py
  run_belief_rho_sweep.py
  run_belief_rho_multiseed.py
  run_spectral_sweep.py
  check_consensus_mode.py
  run_null_topology_test.py
  spectral_diagnostics.py
  make_paper_plots.py
```

Generated outputs are written to:

```text
results/
```

The `results/` directory is ignored by git. Figures and CSVs should be regenerated from the scripts.

## Setup

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

If you do not want to install the package in editable mode, use:

```bash
PYTHONPATH=src python3 -m experiments.make_paper_plots
```

## Reproduce the paper figures

```bash
PYTHONPATH=src python3 -m experiments.make_paper_plots
```

This writes the paper figures to:

```text
results/paper/
```

Expected outputs include:

```text
results/paper/fig1_w1_vs_rho.png
results/paper/fig1_w1_vs_rho.pdf
results/paper/fig2_consensus_alignment.png
results/paper/fig2_consensus_alignment.pdf
```

## Main experiments

Run the rho sweep:

```bash
PYTHONPATH=src python3 -m experiments.run_belief_rho_sweep
```

Run the multi-seed version:

```bash
PYTHONPATH=src python3 -m experiments.run_belief_rho_multiseed
```

Run the spectral diagnostics:

```bash
PYTHONPATH=src python3 -m experiments.run_spectral_sweep
```

Check direct consensus-mode alignment:

```bash
PYTHONPATH=src python3 -m experiments.check_consensus_mode
```

Run the null topology test:

```bash
PYTHONPATH=src python3 -m experiments.run_null_topology_test
```

## How to read the experiments

The important comparison is not just:

```text
linear vs star vs small-world
```

The important comparison is:

```text
topology-induced variation
vs
noise-realization-induced variation
```

If redrawing the forecast noise changes the dominant-mode fraction more than changing the graph, then topology is below the observable noise floor for this diagnostic.

That is the Semele effect.

## What this project is not

This is not a full grid model.

It does not include:

- AC power flow,
- market clearing,
- distribution constraints,
- transformer thermal dynamics,
- real Cyprus grid data,
- or real battery fleet telemetry.

The simulator is intentionally small. Its purpose is to isolate one mechanism:

```text
shared stochastic forcing can dominate topology-dependent collective dynamics
```

## Status

Research prototype.

The code is intended to reproduce the experiments and figures behind the Semele note, not to serve as a production-grade battery dispatch simulator.

## License

See `LICENSE`.
