from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.battery import Battery


@dataclass
class SimulationResult:
    """Aggregate outcomes from one simulation run."""

    aggregate_load_kw: np.ndarray
    aggregate_battery_power_kw: np.ndarray
    feeder_violations: np.ndarray  # bool array, True when feeder load > feeder_limit


def run_simulation(
    load_profiles_kw: np.ndarray,
    prices: np.ndarray,
    batteries: list[Battery],
    controller,
    dt_hours: float = 1.0,
    low_threshold: float = 30.0,
    high_threshold: float = 70.0,
    feeder_limit_kw: float = 60.0,
    neighborhood_size: int = 2,
) -> SimulationResult:
    """
    Simulate one battery fleet under a shared controller.

    Decision model — simultaneous:
        All batteries observe the same base feeder load (sum of raw load profiles
        at this timestep) before any battery has acted. This avoids ordering
        effects and reflects agents deciding in parallel.

    Neighbourhood signal:
        Each battery receives the average actual power of its k nearest
        neighbours from the previous timestep. At t=0 this is zero for all.
        Controllers that do not use this signal ignore it via **_ignored.

    All available context is passed to every controller call.
    Each controller takes what it needs and ignores the rest.
    """

    n_steps, n_batteries = load_profiles_kw.shape

    aggregate_load       = np.zeros(n_steps)
    aggregate_battery_kw = np.zeros(n_steps)
    feeder_violations    = np.zeros(n_steps, dtype=bool)

    # Neighbourhood memory: actual power delivered last timestep, per battery.
    # Positive = discharged, negative = charged.
    prev_actions = np.zeros(n_batteries)

    for t in range(n_steps):
        base_load_kw = float(load_profiles_kw[t].sum())
        battery_powers: list[float] = []

        for i, battery in enumerate(batteries):

            # Neighbourhood signal: average of k nearest neighbours' last actions.
            neighbour_idx = [(i + j) % n_batteries for j in range(1, neighborhood_size + 1)]
            neighbor_avg_kw = float(np.mean(prev_actions[neighbour_idx]))

            requested_kw = controller(
                price           = float(prices[t]),
                low_threshold   = low_threshold,
                high_threshold  = high_threshold,
                max_power_kw    = battery.max_charge_kw,
                soc             = battery.soc,
                feeder_load_kw  = base_load_kw,
                feeder_limit_kw = feeder_limit_kw,
                neighbor_avg_kw = neighbor_avg_kw,
            )

            actual_kw = battery.apply_power(requested_kw, dt_hours)
            battery_powers.append(actual_kw)

        prev_actions = np.array(battery_powers)

        total_battery_kw = float(sum(battery_powers))

        # Positive battery power = discharging → reduces feeder demand.
        # Negative battery power = charging   → increases feeder demand.
        feeder_load_kw = base_load_kw - total_battery_kw

        aggregate_load[t]       = feeder_load_kw
        aggregate_battery_kw[t] = total_battery_kw
        feeder_violations[t]    = feeder_load_kw > feeder_limit_kw

    return SimulationResult(
        aggregate_load_kw        = aggregate_load,
        aggregate_battery_power_kw = aggregate_battery_kw,
        feeder_violations        = feeder_violations,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_peak_kw(aggregate_load_kw: np.ndarray) -> float:
    """Highest feeder load observed in the simulation."""
    return float(np.max(aggregate_load_kw))


def compute_max_ramp_kw(aggregate_load_kw: np.ndarray) -> float:
    """Largest single-step change in feeder load."""
    if len(aggregate_load_kw) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(aggregate_load_kw))))


def count_feeder_violations(feeder_violations: np.ndarray) -> int:
    """Number of timesteps where feeder load exceeded the limit."""
    return int(np.sum(feeder_violations))


def compute_total_feeder_excess_kw(
    aggregate_load_kw: np.ndarray,
    feeder_limit_kw: float,
) -> float:
    """Cumulative load above the feeder limit across all timesteps."""
    return float(np.sum(np.maximum(0.0, aggregate_load_kw - feeder_limit_kw)))