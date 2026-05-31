# Battery Coordination Sandbox

## What this is

A simulation study of how multiple batteries sharing a feeder create aggregate behavior that no individual battery intended.

The central question:

> When many batteries follow simple local rules, what happens to the grid?

This is not a grid tool. It is an experimental sandbox for studying coordination failure.

---

## What Phase 1 showed

Four controllers were compared across multiple system regimes (light, medium, hard feeder stress):

- **TOU** — naive local dispatch. Causes strong synchronization and the worst feeder peaks.
- **Randomized TOU** — breaks synchronization partially, but effect is fragile and regime-dependent.
- **Hard Capped TOU** — suppresses overloads abruptly. Best for peak and ramp in hard regimes, but rigid.
- **Soft Capped TOU** — scales battery response continuously with feeder stress. Most balanced overall.

### Key findings

1. Local rationality does not scale. Rules that seem harmless for one battery can be harmful for many.
2. Controller performance is regime-dependent. No single strategy dominates across all conditions.
3. Soft coordination outperforms randomization because its parameter has structured, predictable effects.
4. In hard regimes, simple controllers do not eliminate failure — they reshape how the system fails.
5. Violation count is a misleading metric. Total excess is more informative under stress.

---

## What Phase 2 showed

**Question:** How much information does an agent need before its local behavior stops being harmful?

Four information levels were tested, each mapped to a controller:

- **None** — price signal only (TOU)
- **Local** — own state only (SoC-proportional scaling)
- **Neighborhood** — average of k neighbours' previous actions
- **Global** — full feeder load (soft capped TOU)

Tested across a 2×2 robustness design:

|                       | Deterministic prices | Stochastic prices |
|-----------------------|---------------------|-------------------|
| **Uniform SoC**       | ✓                   | ✓                 |
| **Heterogeneous SoC** | ✓                   | ✓                 |

Neighborhood size was additionally swept from 1 to n−1 (full fleet visibility).

### Key findings

1. **Neighborhood information provides zero benefit at every scale.** Peak reduction was 0% of baseline across all regimes, all configurations, and all neighborhood sizes from 1 to 19. The failure is not about how many peers you observe — it is about the nature of the signal. Lagged peer actions do not approximate real-time feeder state regardless of fleet coverage.

2. **Local self-awareness reduces peak by ~25% under predictable prices, ~13% under stochastic prices.** The SoC-proportional mechanism works by natural self-limiting — batteries that are nearly full charge less aggressively. Effectiveness depends on predictable price structure to build consistent SoC patterns.

3. **Global feeder visibility is robust to price uncertainty and gets stronger under stochastic prices.** Peak reduction holds at 40–46% in medium and hard regimes regardless of price structure, because the controller responds directly to system stress rather than inferring it through price signals.

4. **The type of information matters more than the quantity.** Direct measurement of the system variable you care about (feeder load) is more robust than indirect inference through individual state (SoC). Neighbourhood information — neither direct nor individual — adds nothing.

5. **The original hypothesis was wrong in an interesting way.** Most benefit does not come from minimal information sharing. It comes from the right kind of information. Minimal sharing (neighbourhood) failed entirely; individual state (local) gave partial benefit; only system-level visibility (global) gave consistent results.

---

## Structure

```
src/              battery, controllers, simulator
experiments/      basic_run.py, information_sweep.py, neighborhood_size_sweep.py
notebooks/        coordination_comparison.ipynb
results/          csv outputs per configuration
```

## Run

```bash
pip install -r requirements.txt

# Phase 1
python3 experiments/basic_run.py
jupyter notebook notebooks/coordination_comparison.ipynb

# Phase 2
python3 experiments/information_sweep.py
python3 experiments/neighborhood_size_sweep.py
```