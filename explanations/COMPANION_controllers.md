# controllers.py — block by block

**What it does:** the rules each battery uses to decide how hard to charge or discharge.

---

## What was removed, and why
- `randomized_tou_controller`, `capped_controller`, `capped_tou_controller`, `soft_capped_tou_controller`, `aggregate_requests`, `local_aware_controller`, `neighborhood_controller`, and `droop_controller`.
- None of the paper experiments use them. Only `experiments/legacy/` did, and that folder is in `.gitignore`, so it isn't published anyway.
- The step-4 mitigations will be added back as new, clean functions.

## The `**_ignored` trick
- The simulator passes every controller the same long list of inputs.
- `**_ignored` lets a controller silently drop the ones it doesn't need.

## `tou_controller`
TOU stands for time of use.
- Price at or below the low threshold: charge at full power. The function returns −max.
- Price at or above the high threshold: discharge at full power. It returns +max.
- Otherwise: do nothing.

## `belief_neighborhood_controller` (the paper's controller)
1. **Start from the TOU request.** If the request is to discharge or to idle, return it unchanged. Only charging stresses the feeder.
2. **Clamp `alpha`** (`belief_weight`) to the range 0–1.
3. **`feeder_scale`** = 1 − feeder load ÷ limit, clipped to 0–1.
   - An empty feeder gives 1, so charge fully.
   - A feeder at its limit gives 0, so don't charge.
   - In community mode the feeder load is load minus solar. It is often negative at midday, which clips to 1, so the battery charges fully.
4. **`neighbor_scale`** = 1 − how hard the neighbours charged last hour, measured as a fraction of full power and clipped to 0–1.
5. **Final request** = charge × (alpha × feeder_scale + (1 − alpha) × neighbor_scale).
   - The neighbour term is the only place the network shape matters.
