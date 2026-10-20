# run_community.py — block by block (new file: steps 1–3)

**Run it from the repo root:** `python -m experiments.run_community`. It takes about 1 minute.

---

## Docstring
- States the three steps and lists the output files.

## Imports
- It reuses the paper's synthetic world (`make_cyprus_style_pv_profile`, `make_load_profiles`, `make_batteries`), the PV accounting (`compute_pv_metrics`), and `spectral_diagnostics` (w1 and PR).
- So the community runs live in the same world as the paper.

## Configuration
- **`RHO_VALUES`** (step 1): 0 as a reference, then 0.7 to 1.0, the realistic range for houses under the same clouds.
- **`PRICE_RULES`** (step 2): `fixed` vs `sdr`.
- **`BATTERY_POWER_KW`** (step 3): 3, 5, 7 and 10 kW per house, against a feeder of 6 kW per house.
- **`BATTERY_HOURS = 4`**: capacity is power × 4 hours, so a 5 kW battery holds 20 kWh, as in the paper.
- The prices, thresholds and noise sizes are the paper's values. The grid price is 80 and the feed-in price is 20.
- **Topology:** only small-world. The paper already showed that topology leaves no fingerprint, so running three graphs would triple the time for nothing.

## `one_run`
- **`rng = default_rng(SEED + run_id)`**: run 7 gets the same houses, loads and starting charge under *every* condition. Only the thing being tested changes. This is called common random numbers, and it makes the comparisons much less noisy.
- **Batteries:** it takes the random starting charge from `make_batteries`, then builds batteries of the size being tested.
- **`np.random.seed(...)`**: the same noise draws for every condition.
- **`run_simulation(...)`**: community mode. Solar goes into the feeder, and the price rule is the one being tested.
- **`net0` / `viol0`**: the same day with no batteries. `battery_caused_viol` = violations with batteries − violations without, so only the damage the batteries do is counted.
- **Metrics per run:**
  - `peak_charge_ratio`: the biggest total charging power, divided by the feeder limit.
  - `import_viol` / `export_viol`: the share of hours where each limit is broken.
  - `curtailment`: the share of surplus solar that no battery absorbed.
  - `w1`, `PR`: how much the fleet moves as one, the paper's measure.
  - `price_std`: how much the price moves during the day.
  - `price_flips`: the share of hours where the price jumps by more than half its range (more than 30 here). A high value means the price flip-flops.

## `gates`
- **Gate A (step 2):** at ρ = 1, compare fixed vs sdr on peak, battery-caused violations, and price flips. It says "amplifies" if sdr adds more than 0.01 to violations and "damps" if it removes more than 0.01.
- **Gate B (step 3):** the smallest battery size where charging alone breaks the feeder, for each rule.

## `plot`
- Four panels against ρ: peak, battery-caused violations, curtailment, and w1.
- Colour = battery size. Solid line = fixed price. Dashed line = sdr.

## `main`
1. Builds the solar profile and the fixed prices.
2. Loops over every combination: 4 sizes × 2 rules × 8 ρ values × 100 runs = 6400 runs.
3. Saves the raw table, averages per condition, writes the gates to `gates.txt`, and saves the plot.
