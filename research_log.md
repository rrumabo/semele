Phase 1:
Finding: local rationality does not scale.
Mechanism: synchronized TOU charging creates feeder peaks
no individual battery intended.
Soft coordination reshapes failure rather than eliminating it.
Regime matters — no single controller dominates across all conditions.

Phase 2 — information gradient:
Finding: information type matters more than quantity.
Neighborhood (lagged peer signals) → zero benefit at every scale.
Local (own SoC) → ~25% peak reduction, weakens under stochastic prices.
Global (feeder load) → robust, strengthens under stochastic prices.
Reason: direct measurement of the variable you care about
outperforms indirect inference regardless of how much you share.

Phase 3 — fairness_sweep :
Finding: optimal aggregate outcomes do not imply fair distribution.
Same battery, same desire, only position differs.
End-of-line bears 3.4× its fair share of curtailment.
Hard regime: end-of-line batteries charged nothing.
No malice. Pure mechanism.

Note on scope:
This simulation is complex but tractable — deterministic agents,
smooth controllers, reproducible results.
Adaptive and memory-based agents are deliberately excluded.
Sensitivity to initial conditions and attractor identification
are open questions for future work.

Phase 4A: frequency_sweep
Question: what is the stability boundary in M × droop_gain space?
Method: sweep M=[6-50] × droop_gain=[0.5-10].
Finding: stability condition is approximately M/droop_gain ≥ 5.
Below this ratio: frequency diverges (hits ±5 Hz clamp).
Above this ratio: frequency deviation stays within 1 Hz of nominal.
Best stable config: M=50, droop=2 → std=0.053 Hz.

Phase 4B:
M=0 with droop is not well defined — division by zero in the swing equation.
Droop requires inertia to exist. They are not alternatives.

Phase 4 — disturbance sweep:
Stability boundary robust: droop ≤ 2 stable at M=10
regardless of disturbance size (5–25 kW).
Optimal droop: 1.0 — consistently lowest frequency std.
Timing matters: earlier disturbance → worse nadir,
independent of disturbance size.
Anomaly: disturbance=20 kW at t11 pushed droop=0.5
to near-instability (nadir=45.07 Hz).
New variable identified: drop timing interacts with droop gain.

Phase 5 — Mixed fleet:
Stability threshold: 60% droop fraction (6/10 batteries).
Near-source-first = random assignment — both stabilize at 60%.
End-of-line-first needs 70% — position matters at the margin.
Frequency is a global signal — droop effectiveness is
position-independent, unlike curtailment burden (Phase 3).
std improvement is smooth — each additional droop agent helps equally.

Phase 6 A — topology sweep:
Expected: disorder → order as k increases (Kuramoto-style).
Found: quasi-order → brief sync window → polarization.

Sync window:
  small_world: k=[0.05, 0.21], max 90% of runs
  linear:      k=[0.05, 0.41], max 75% of runs
  star:        never syncs

Polarization onset:
  small_world: k>0.31 (faster due to shortcuts)
  linear:      k>0.46
  star:        never — hub mediation prevents extremes

Finding: small-world accelerates BOTH synchronization AND polarization.
Shortcuts are a double-edged sword.

Phase 6 B — heterogeneous k:
Leader emergence confirmed: corr(k_i, |x_i|) = 0.84-0.86 for bimodal.
Agents with k∈[1.0,1.5] reach |x|≈1.0, followers partially dragged.

Small-world: best for homogeneous k (sync=70%), worst for heterogeneous (complete collapse).
Star: most robust — hub dampens leader influence.

Key finding: topology determines HOW leaders affect the system,
not WHETHER leaders exist.

Phase 7A — rho_agents sweep: correlated price-signal noise

Question:
What is the critical belief correlation rho_c above which decentralized
battery agents lose action diversity and synchronize?

Setup:
  30 batteries
  tou_controller
  price_sigma = 20.0
  forecast_error_sigma_kw = 1.0
  rho_agents swept from 0 to 1
  40 rho points
  20 runs per point
  4 topologies: legacy_ring, linear, small_world, star

Finding — rho_c_sync_collapse ordering:
  legacy_ring   0.205
  small_world   0.205
  linear        0.231
  star          0.308

Correlated price-signal noise causes gradual synchronization collapse.
The decline is not a sharp cliff — it is a smooth common-noise collapse,
consistent with common-noise synchronization literature (Pimenova et al. 2016).

Topology modulates rho_c. Star resists collapse longest. Legacy_ring and
small_world collapse earliest. Linear is slightly later.

Spectral conjecture test:
lambda_2 alone does not explain rho_c.
Measured lambda_2:
  linear       0.0979
  star         1.0000
  small_world  1.4110
  legacy_ring  1.7639

Linear and legacy_ring share nearly identical rho_c despite an 18-fold
difference in lambda_2. rho_c is not predicted by algebraic connectivity alone.
Full spectral structure, eigenvector localization, and degree asymmetry
are needed to explain the ordering.

Shape vs shift:
Topology changes both the position (rho_c) and the slope of the collapse curve.
  legacy_ring   slope 0.755  (gentlest)
  small_world   slope 0.998
  linear        slope 1.147
  star          slope 1.221  (steepest)
More resistant topologies collapse more steeply once triggered.
Total collapse magnitude (~82%) is topology-independent.

rho_c_failure: not measured in this sub-experiment.
See Phase 7B.

Phase 7b — feeder_failure sweep: belief_neighborhood_controller
Question:
At what rho does synchronised charging cause feeder overload — and
how large is the protection window between collapse and failure?
Setup:
10 batteries
belief_neighborhood_controller
feeder_limit_kw = 60.0
price_sigma = 20.0
rho_agents swept 0 to 1
40 rho points, 20 runs per point
4 topologies
Baseline violations at rho=0: 0.01–0.03 (clean)
Finding — rho_c and protection windows:
  topology      rho_c_sync  rho_c_failure  window
  small_world   0.282       0.487          0.205
  linear        0.308       0.359          0.051
  legacy_ring   0.333       0.462          0.128
  star          0.410       0.590          0.179
Key findings:
Collapse onset and feeder failure are independent properties.
Small_world collapses earliest but has the largest protection window.
Linear collapses second but has the smallest protection window.
Spectral conjecture falsified: lambda_2 does not predict rho_c.
Local degree structure and information aggregation govern rho_c,
not global spectral mixing rate.
Delay-but-amplify pattern confirmed: topologies with higher rho_c
collapse more steeply once triggered (star slope 1.22, legacy_ring 0.76).

Phase 8 — Fleet size scaling

Question:
Does fleet size change the feeder impact of correlated belief when feeder capacity scales proportionally with fleet size?

Setup:
Linear topology only.
Fleet sizes N = 5, 10, 20, 30, 50.
feeder_limit_kw = 6N.
belief_neighborhood_controller.
price_sigma = 20.
rho_agents swept from 0 to 1.

Finding:
Baseline violation rates remain low across fleet sizes, confirming that feeder headroom scaling prevents trivial overload at larger N.

However, high-correlation violation rates grow dramatically relative to baseline. The violation amplification ratio increases with fleet size. This suggests that larger fleets can appear safer under independent noise while becoming more exposed to common-noise synchronization.

Threshold caveat:
The previous baseline-relative 2σ rho_c_failure detector is not comparable across fleet sizes, because baseline variance shrinks with N. This can create artificial early failure thresholds and negative protection windows. Therefore Phase 8 reports violation amplification rather than protection_gap.

Conclusion:
Fleet size does not simply make the feeder fail earlier under proportional headroom. Instead, scale increases the contrast between normal decentralized operation and correlated synchronized failure.

Phase 9 — failure threshold calibration

The baseline-relative 2σ failure detector is not comparable across fleet sizes because baseline variance changes with N. At larger N, independent load/control variation averages out, making baseline feeder violations smoother and lowering the effective failure threshold. This can produce artificial early rho_c_failure estimates.

A fixed absolute violation threshold is more comparable across fleet sizes. Using violation_mean > 0.10, rho_c_failure remains in the range ~0.63–0.84 across N, showing that proportional feeder scaling prevents trivial earlier failure at larger fleet sizes.

However, violation amplification increases with fleet size. At rho≈0.9, violations rise by ~2.3x for N=5 but ~7.9x for N=50. At rho=1.0, violations rise by ~3.1x for N=5 and ~9.2x for N=50.

Conclusion:
With feeder capacity scaled proportionally, larger fleets do not necessarily fail earlier in absolute rho. Instead, they become more deceptive: baseline operation appears safer and more stable, while common-noise synchronization produces a larger relative jump in feeder stress.