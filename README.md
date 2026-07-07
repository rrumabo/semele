# Semele

Semele is a small computational study of decentralized battery-fleet dynamics under shared stochastic forecast signals.

The core result is that, when agents are exposed to correlated forecast/price errors, the observed effective dimensionality of the fleet trajectory can become dominated by the common stochastic component rather than by the communication topology. In the linearized model, the correlated component projects onto the graph-invariant consensus mode; topology acts only through transverse modes.

This repository contains the simulation code and experiments used for the note:

**The Semele Effect: When Shared Stochastic Forcing Overwhelms Topological Structure**

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
  legacy/
```

# Generated outputs are written to results/, which is ignored by git.

# Setup
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

If running without editable install, use: PYTHONPATH=src python3 -m experiments.make_paper_plots

Reproduce paper figures : PYTHONPATH=src python3 -m experiments.make_paper_plots

## Main experiments
PYTHONPATH=src python3 -m experiments.run_belief_rho_sweep
PYTHONPATH=src python3 -m experiments.run_belief_rho_multiseed
PYTHONPATH=src python3 -m experiments.run_spectral_sweep
PYTHONPATH=src python3 -m experiments.check_consensus_mode
PYTHONPATH=src python3 -m experiments.run_null_topology_test

# Claim

The claim is not that topology has no dynamical effect. The claim is narrower: under shared stochastic forcing, topology-dependent modes can be masked because the dominant variance is injected into the consensus mode, which every connected graph shares.