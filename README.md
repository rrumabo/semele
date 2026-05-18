# Battery Coordination Sandbox — Aggregate Grid Behavior Under Local Control

## Overview

This project studies how multiple batteries interacting on the same feeder create aggregate grid behavior.

The central question is:

> If many batteries follow simple local charging rules, what happens to the feeder?

A strategy that looks reasonable for one battery may become harmful when many batteries act together.

This project explores that coordination problem through simulation.

It compares multiple controller strategies under identical operating conditions and evaluates how aggregate behavior changes across system regimes.

This is not a production grid tool.

It is an experimental coordination sandbox.

---

## Core Problem

Distributed energy assets often make local decisions:

- charge when electricity is cheap
- charge when surplus exists
- react independently

Individually, these rules seem sensible.

Collectively, they can create:

- synchronized peaks
- feeder overloads
- large ramp events
- unstable aggregate behavior

This project investigates that failure mode.

---

## System Model

The simulated system contains:

- multiple batteries
- shared feeder constraint
- aggregate demand signal
- controller coordination logic

Each battery has:

- energy capacity
- charge/discharge power limits
- state-of-charge dynamics

The focus is intentionally simplified:

the goal is to understand coordination behavior, not replicate full grid physics.

---

## Controllers Compared

### 1. Time-of-Use (TOU)

Naive local strategy:

- batteries charge according to predefined timing logic
- no awareness of aggregate feeder stress

Behavior:
- simple
- uncoordinated
- synchronization-prone

---

### 2. Randomized TOU

Adds randomness to reduce synchronization.

Idea:

instead of all batteries acting identically, stagger behavior.

Behavior:
- partially coordinated
- reduced synchronization
- inconsistent performance

---

### 3. Hard Capped TOU

Introduces explicit feeder constraint behavior.

Idea:

prevent charging when feeder stress exceeds threshold.

Behavior:
- stronger overload protection
- rigid controller response
- can create tradeoffs elsewhere

---

### 4. Soft Capped TOU

Continuous coordination strategy.

Idea:

gradually reduce charging pressure as feeder stress increases.

Behavior:
- smoother adaptation
- less abrupt behavior
- improved coordination under some regimes

---

## Metrics

Controllers are compared using:

### Grid metrics
- feeder peak load
- ramp magnitude
- number of violations
- total overload severity

These measure system stress.

---

## Experiments

The project includes:

### Baseline comparison
Direct controller comparison under one scenario.

---

### Regime sweeps
Systematically varying:

- number of batteries
- feeder limits

to study scaling behavior.

---

### Softness sensitivity
Testing how soft coordination tuning changes outcomes.

---

### Randomness sensitivity
Testing how synchronization disruption changes outcomes.

---

## How to Run

1. Install dependencies:
pip install -r requirements.txt
2. Run baseline experiments:
python3 experiments/basic_run.py
3. Open notebook for analysis:
jupyter notebook
4. Then open:
notebooks/coordination_comparison.ipynb

---

## Key Findings

Observed themes:

### Local logic does not scale automatically
Rules that seem harmless for one battery can create harmful aggregate behavior.

---

### Coordination quality is regime-dependent
A controller that performs well in light stress may degrade under harder conditions.

There is no universally best strategy.

---

### Randomness helps, but inconsistently
Breaking synchronization can reduce some stress metrics but does not guarantee robust performance.

---

### Soft coordination can outperform rigid logic
Continuous adaptation often improves aggregate behavior compared to abrupt threshold rules.

---

## Repository Structure

```text
src/
  battery.py
  controllers.py
  simulator.py

experiments/
  basic_run.py

notebooks/
  coordination_comparison.ipynb

results/
  basic_run_results.csv
  softness_sweep_results.csv
  randomness_sweep_results.csv

--- 

## Future Direction 

Planned extensions include:

* EV charging demand
* PV generation
* multi-asset coordination
* uncertainty-aware strategies
* richer distributed energy resource coordination experiments