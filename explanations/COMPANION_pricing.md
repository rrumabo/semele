# pricing.py — block by block (new file)

**What it does:** holds the two price rules, so the simulator and the experiments all use the same ones.

---

## `pv_shaped_price`
This is the paper's rule, moved here from `reproduce_core_result.py`.
- `pv_norm` is solar divided by the peak solar of the day, so it runs from 0 to 1.
- price = `high − swing × pv_norm`. With the defaults (80 and 60), that is 80 at night and 20 at the solar peak.
- The price is decided in advance. Nothing the batteries do changes it.

## `sdr_price`
SDR stands for supply–demand ratio. The price is set by the street itself, every hour.
- **Inputs:**
  - `supply`: total surplus of the houses that have extra.
  - `demand`: total shortfall of the houses that need power.
  - `grid_price`: what the grid charges. This is the ceiling.
  - `feed_in_price`: what the grid pays for exports. This is the floor.
- **First check:** if nobody needs power (`demand ≤ 0`), the price is the floor.
- **`sdr = supply / demand`.**
  - `sdr ≥ 1`: local surplus covers everyone, so the price is the floor.
  - `sdr = 0`: no local surplus, so the price is the ceiling.
  - In between, the price slides smoothly between the two.
- **The two formula lines** come from Liu et al., *IEEE Trans. Power Systems* 2017.
  - `sell` is the price sellers get.
  - The returned value is what buyers pay: part local, part grid, weighted by `sdr`.
- **Worked check**, with ceiling 80 and floor 20:

| supply | demand | price |
|---|---|---|
| 0 | 10 | 80.00 |
| 2.5 | 10 | 71.43 |
| 5 | 10 | 56.00 |
| 10 | 10 | 20.00 |

- **Assumption to verify:** I wrote the formula from my knowledge of the Liu et al. paper. Check it against the paper before you cite it.
