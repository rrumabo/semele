# simulator.py — block by block

**What the file does:** runs one day (24 hourly steps) of N home batteries behind one shared feeder cable.

**Sign rule used everywhere:** battery power below 0 = charging, above 0 = discharging.

---

## What was removed, and why
- **Frequency model** (`M`, `omega`, `omega_nominal`, and the frequency metric helpers). It was switched off in every paper run (`M = 0`), so it never did anything.
- **`positions`, `position_alpha`.** Unused by every experiment.
- **`neighborhood_size` fallback.** Every experiment passes a network, so the fallback never ran.
- **`generation_kw`.** It only fed the frequency model.
- **Per-battery curtailment tracking.** No experiment read it.
- **The phase 1–3 metric helpers.** No experiment called them.

**Check:** after the cleanup, `reproduce_core_result` and all six paper scripts give identical numbers. The differences are at most 1e-13, which is floating-point noise.

---

## Imports
- `dataclass` is a shortcut for a class that only holds data.
- `numpy` handles arrays.
- `Battery`, `Network`, and `sdr_price` are the three Semele parts the simulator uses.

## `SimulationResult`
A box holding the outputs of one run.
- `per_battery_power_kw`: a table with one row per hour and one column per battery.
- `aggregate_battery_power_kw`: the sum over all batteries, for each hour.
- `feeder_net_kw`: what flows through the feeder cable. A positive value means the street imports from the grid; a negative value means it exports.
- `feeder_violations`: True in any hour where either limit is broken.
- `import_violations` / `export_violations`: the two limits checked separately.
- `price`: the true price announced each hour, before any noise is added.

## `run_simulation` — the arguments
- **Kept from before:** loads, prices, batteries, controller, network, the price thresholds, the feeder limit, `rho_agents`, and the two noise sizes.
- **New:**
  - `pv_kw`: total solar for the street, one value per hour. If you pass it, the feeder counts solar.
  - `export_limit_kw`: the limit on exporting. It defaults to the same value as the import limit.
  - `price_rule`: either `"fixed"` (the paper's rule) or `"sdr"` (a price set by the community itself).
  - `grid_price` / `feed_in_price`: the most and the least the local price can be.
- **Defaults** are exactly paper mode, so old scripts behave the same.

## Input checks
- `rho_agents` must lie between 0 and 1.
- `price_rule` must be one of the two names.
- `"fixed"` needs a list of prices.
- `"sdr"` needs solar, because it computes surplus.

## Setup lines
- `controllers`: if you pass one controller, every battery uses it. If you pass a list, each battery gets its own.
- `pv`: if no solar is given, it is zero every hour. That reproduces paper mode.
- `pv_per_house`: the street's solar split equally across the houses.
- `export_limit`: falls back to the import limit if none was given.

## The noise block
This is the paper's equation: `noise = sqrt(rho)*Z + sqrt(1-rho)*eps`.
- `Z` is one shock per hour, shared by everyone. Think of it as the same cloud over the street.
- `eps` is a separate shock for each house.
- The random numbers are drawn in the same order as before, which is why the paper's results are unchanged.

## The hourly loop
1. **Price for this hour.**
   - `"fixed"`: read it from the list.
   - `"sdr"`: work out each house's net (load − its solar − what its battery did *last hour*).
     - Houses with a negative net have surplus. Their total is `supply`.
     - Houses with a positive net are short. Their total is `demand`.
     - The price comes from supply ÷ demand (see `pricing.py`).
   - Using *last hour's* battery actions is the feedback loop: what the batteries did changes the price they see next.
2. **Each battery decides.**
   - It looks at the price plus its own noise, scaled by `price_sigma`.
   - It looks at the feeder load: load − solar. In paper mode solar is zero, so this is just the load, as before.
   - It looks at the average of what its neighbours did last hour.
   - The controller turns these into a request, and `battery.apply_power` clips the request to what the battery can physically do.
   - `agent_id` and `t` are also passed to the controller. Current controllers ignore them; they are there for the step-4 mitigations (staggering by house and by hour).
3. **The feeder after the batteries act.** Load − solar − battery power. Charging raises the import. Charging while there is surplus solar cancels export instead, which is the physics the old model missed.

## The violation lines
- Import violation: net above the limit.
- Export violation: net below minus the export limit. This is only checked when solar is present, so paper mode is unchanged.
