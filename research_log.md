Neighborhood result: lagged signals create oscillations, not coordination.
Peak same as baseline — synchronization preserved.
Ramp worse — flip-flop pattern introduced.
Partial design artifact (identical batteries + aggressive scaling), but
the oscillation mechanism is real.

Local result: genuine. Mechanism is demand reduction through SoC-proportional
scaling, not desynchronization. Peak ~25% lower across all regimes.

Local and global diverge under price uncertainty. With predictable prices they're comparable. With stochastic prices, global stays effective while local weakens. The difference is what each controller is actually measuring — own state vs system state. Direct measurement of what you care about (feeder load) is more robust than indirect inference through individual state (SoC).

Neighborhood size sweep: 1, 2, 5, 10, n-1 tested across all regimes.
Peak reduction = 0% at every size. Finding confirmed.
The failure is not about quantity of neighbors — it is about
the nature of the signal. Lagged peer actions do not approximate
real-time feeder state regardless of how many peers you observe.