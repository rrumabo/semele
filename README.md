# Lampyris

Lampyris is a small simulation sandbox for studying coordination failure in distributed battery fleets on renewable-heavy island grids.

It asks a simple question:

> When do distributed batteries help absorb PV surplus, and when do shared forecast or price signals synchronize them into grid-level stress?

This is not a production grid tool. It is an experimental research sandbox.

## Core idea

A single battery responding to a price or forecast signal is harmless.

Many batteries responding to the same signal can become a new system-level load.

Lampyris studies this failure mode by sweeping `rho_agents`, the amount of shared signal correlation across the fleet.

Low `rho_agents` means each battery mostly sees its own private noisy signal.

High `rho_agents` means many batteries see the same common signal.

The core mechanism is:

rho_agents increases
- battery actions become more synchronized
- aggregate charging peaks rise
- feeder violations appear
- PV curtailment-risk proxy increases

The main result is not “batteries are bad”.

The result is:

> Distributed batteries are useful flexibility until shared signals make them act like one coordinated stress event.

## Reproduce the core result

Install the package in editable mode:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

Run the main experiment:

    python3 experiments/reproduce_core_result.py

This generates:

    results/core_rho_sweep.csv
    results/core_summary.csv
    results/core_summary.json
    results/core_rho_sweep.png

## Project structure

    src/lampyris/
        battery.py
        controllers.py
        network.py
        simulator.py
        time_integrators.py

    experiments/
        reproduce_core_result.py
        legacy/

    results/
        core_rho_sweep.csv
        core_summary.csv
        core_summary.json
        core_rho_sweep.png
        legacy/

The official reproducible experiment is:

    experiments/reproduce_core_result.py

Older exploratory experiments are kept in:

    experiments/legacy/

Older exploratory results are kept in:

    results/legacy/

## Current default experiment

The current core run uses:

    N_BATTERIES = 20
    N_RUNS = 200
    FEEDER_LIMIT_PER_BATTERY_KW = 6.0
    PV_PEAK_PER_BATTERY_KW = 5.5

For a quick smoke test, temporarily reduce:

    N_RUNS = 5

Then restore it before regenerating final results.

## Sign convention

Lampyris uses this convention:

    battery_power_kw < 0 = charging
    battery_power_kw > 0 = discharging

Negative power means the battery is consuming from the grid.

Positive power means the battery is supplying power to the load.

## Main metrics

The core experiment tracks:

    sync_index
    peak_charge_ratio
    feeder_violation_fraction
    pv_curtailment_fraction
    pv_absorption_fraction

Interpretation:

    Lower sync_index
        means batteries are acting more similarly.

    Higher peak_charge_ratio
        means aggregate charging pressure is larger relative to the feeder limit.

    Higher feeder_violation_fraction
        means the feeder limit is exceeded more often.

    Higher pv_curtailment_fraction
        means the synthetic PV-surplus risk proxy is worse.

## Limitation

The PV curtailment metric is a simple energy-accounting proxy.

It is not a full power-flow model, OPF model, or market dispatch model.

The correct phrase is:

    PV curtailment-risk proxy

not:

    measured real-world curtailment

## Research log

The detailed phase-by-phase development history is in:

    research_log.md

That file contains the older controller tests, topology experiments, correlated-belief sweeps, fleet-size scaling, and threshold calibration notes.

## One-line summary

Lampyris studies when distributed batteries remain useful flexibility and when shared signals synchronize them into a grid-level stress event.