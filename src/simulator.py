from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.battery import Battery


@dataclass
class SimulationResult:
    """Aggregate and per-battery outcomes from one simulation run."""

    aggregate_load_kw:           np.ndarray
    aggregate_battery_power_kw:  np.ndarray
    feeder_violations:           np.ndarray
    per_battery_curtailment_kwh: np.ndarray
    per_battery_requested_kwh:   np.ndarray
    omega_hz:                    np.ndarray  # grid frequency at each timestep (Phase 4)


def run_simulation(
    load_profiles_kw:  np.ndarray,
    prices:            np.ndarray,
    batteries:         list[Battery],
    controller,
    dt_hours:          float = 1.0,
    low_threshold:     float = 30.0,
    high_threshold:    float = 70.0,
    feeder_limit_kw:   float = 60.0,
    neighborhood_size: int   = 2,
    positions:         list[float] | None = None,
    position_alpha:    float = 0.5,
    # --- Phase 4: frequency dynamics ---
    # M=0 disables frequency tracking — all Phase 1/2/3 experiments unchanged.
    M:             float = 0.0,   # inertia constant (s). 0 = disabled.
    omega_nominal: float = 50.0,  # nominal grid frequency (Hz)
    generation_kw: float = 0.0,   # background generation (kW) — fixed per run
) -> SimulationResult:
    """
    Simulate one battery fleet under a shared controller.

    Decision model — simultaneous:
        All batteries observe the same base feeder load before any battery
        has acted. This avoids ordering effects.

    Position model (Model B):
        Each battery has a position in [0, 1].
        0 = near source, 1 = end of line.
        End-of-line batteries have a tighter local feeder limit:
            local_limit_i = feeder_limit x (1 - position_i x position_alpha)
        position_alpha=0.5 means end-of-line limit is 50% tighter than source.

    Curtailment tracking:
        Curtailment is measured against the unconstrained TOU desire —
        what the battery WOULD have done with no feeder awareness.

    Frequency dynamics (Phase 4):
        Enabled when M > 0. Uses the swing equation:
            dω/dt = (1/M) * net_injection
        where net_injection = generation + battery_power - base_load.
        Frequency is updated at the TOP of each timestep so batteries
        can see it before deciding. Imbalance from the previous timestep
        drives the current timestep's frequency (simultaneous decisions).
        When M=0 (default), omega_hz is flat at omega_nominal —
        all existing experiments behave identically to before.
    """

    n_steps, n_batteries = load_profiles_kw.shape
# Normalise generation to per-timestep array regardless of input type
    if np.isscalar(generation_kw):
        generation_array = np.full(n_steps, float(generation_kw))
    else:
        generation_array = np.asarray(generation_kw, dtype=float)

    if positions is None:
        positions = [0.0] * n_batteries

    # --- output arrays ---
    aggregate_load          = np.zeros(n_steps)
    aggregate_battery_kw    = np.zeros(n_steps)
    feeder_violations       = np.zeros(n_steps, dtype=bool)
    per_battery_curtailment = np.zeros(n_batteries)
    per_battery_requested   = np.zeros(n_batteries)
    omega_history           = np.zeros(n_steps)

    # --- state variables ---
    prev_actions    = np.zeros(n_batteries)
    omega           = omega_nominal   # current frequency — starts at nominal
    prev_imbalance  = 0.0

    for t in range(n_steps):

        # --- 1. Update frequency from previous timestep's imbalance ---
        # Batteries will see this omega when deciding what to do.
        if M > 0.0:
            omega += (dt_hours / M) * prev_imbalance
            # Safety clamp: prevent unphysical runaway
            omega = float(np.clip(omega, omega_nominal - 5.0, omega_nominal + 5.0))
        omega_history[t] = omega

        # --- 2. Batteries observe system state and decide ---
        base_load_kw    = float(load_profiles_kw[t].sum())
        battery_powers: list[float] = []

        for i, battery in enumerate(batteries):

            neighbour_idx   = [(i + j) % n_batteries for j in range(1, neighborhood_size + 1)]
            neighbor_avg_kw = float(np.mean(prev_actions[neighbour_idx]))
            local_limit_kw  = feeder_limit_kw * (1.0 - positions[i] * position_alpha)

            requested_kw = controller(
                price           = float(prices[t]),
                low_threshold   = low_threshold,
                high_threshold  = high_threshold,
                max_power_kw    = battery.max_charge_kw,
                soc             = battery.soc,
                feeder_load_kw  = base_load_kw,
                feeder_limit_kw = local_limit_kw,
                neighbor_avg_kw = neighbor_avg_kw,
                omega           = omega,          # Phase 4 — droop uses this
                omega_nominal   = omega_nominal,  # Phase 4 — droop uses this
            )

            actual_kw = battery.apply_power(requested_kw, dt_hours)
            battery_powers.append(actual_kw)

            # --- Curtailment tracking ---
            price_t = float(prices[t])
            if price_t <= low_threshold:
                tou_desire_kw = -battery.max_charge_kw
            elif price_t >= high_threshold:
                tou_desire_kw = battery.max_discharge_kw
            else:
                tou_desire_kw = 0.0

            if tou_desire_kw < 0.0:
                desired_energy   = abs(tou_desire_kw) * dt_hours
                delivered_energy = max(0.0, -actual_kw) * dt_hours
                curtailed_energy = max(0.0, desired_energy - delivered_energy)
                per_battery_requested[i]   += desired_energy
                per_battery_curtailment[i] += curtailed_energy

        # --- 3. Compute aggregate outcomes ---
        prev_actions     = np.array(battery_powers)
        total_battery_kw = float(sum(battery_powers))
        feeder_load_kw   = base_load_kw - total_battery_kw

        aggregate_load[t]       = feeder_load_kw
        aggregate_battery_kw[t] = total_battery_kw
        feeder_violations[t]    = feeder_load_kw > feeder_limit_kw

        # --- 4. Compute imbalance for next timestep's frequency update ---
        # net injection = generation + battery discharge - demand
        # positive = surplus = frequency will rise next step
        # negative = deficit = frequency will fall next step
        prev_imbalance = float(generation_array[t]) + total_battery_kw - base_load_kw

    return SimulationResult(
        aggregate_load_kw           = aggregate_load,
        aggregate_battery_power_kw  = aggregate_battery_kw,
        feeder_violations           = feeder_violations,
        per_battery_curtailment_kwh = per_battery_curtailment,
        per_battery_requested_kwh   = per_battery_requested,
        omega_hz                    = omega_history,
    )


# ---------------------------------------------------------------------------
# Metric helpers — Phases 1–3
# ---------------------------------------------------------------------------

def compute_peak_kw(aggregate_load_kw: np.ndarray) -> float:
    return float(np.max(aggregate_load_kw))


def compute_max_ramp_kw(aggregate_load_kw: np.ndarray) -> float:
    if len(aggregate_load_kw) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(aggregate_load_kw))))


def count_feeder_violations(feeder_violations: np.ndarray) -> int:
    return int(np.sum(feeder_violations))


def compute_total_feeder_excess_kw(
    aggregate_load_kw: np.ndarray,
    feeder_limit_kw:   float,
) -> float:
    return float(np.sum(np.maximum(0.0, aggregate_load_kw - feeder_limit_kw)))


# ---------------------------------------------------------------------------
# Metric helpers — Phase 4 (frequency)
# ---------------------------------------------------------------------------

def compute_frequency_nadir(omega_hz: np.ndarray) -> float:
    """Lowest frequency reached — how deep the disturbance drove it."""
    return float(np.min(omega_hz))


def compute_frequency_peak(omega_hz: np.ndarray) -> float:
    """Highest frequency reached — how far above nominal it went."""
    return float(np.max(omega_hz))


def compute_frequency_recovery_time(
    omega_hz:      np.ndarray,
    omega_nominal: float = 50.0,
    tolerance:     float = 0.1,
    dt_hours:      float = 1.0,
) -> float:
    """
    Time (hours) until frequency returns within tolerance of nominal.
    Returns full simulation duration if recovery never happens.
    """
    recovered = np.where(np.abs(omega_hz - omega_nominal) <= tolerance)[0]
    if len(recovered) == 0:
        return float(len(omega_hz)) * dt_hours
    return float(recovered[0]) * dt_hours


def compute_frequency_std(omega_hz: np.ndarray) -> float:
    """Standard deviation of frequency — measures overall stability."""
    return float(np.std(omega_hz))