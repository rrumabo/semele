"""Semele simulator: one day of a battery fleet behind one shared feeder.

Sign convention everywhere: battery power < 0 = charging, > 0 = discharging.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semele.battery import Battery
from semele.network import Network
from semele.pricing import sdr_price


@dataclass
class SimulationResult:
    per_battery_power_kw:       np.ndarray  # (T, N) power of each battery
    aggregate_battery_power_kw: np.ndarray  # (T,)  sum over batteries
    feeder_net_kw:              np.ndarray  # (T,)  power drawn through the feeder (< 0 = export)
    feeder_violations:          np.ndarray  # (T,)  import OR export limit broken
    import_violations:          np.ndarray  # (T,)
    export_violations:          np.ndarray  # (T,)
    price:                      np.ndarray  # (T,)  true price announced each step


def run_simulation(
    load_profiles_kw: np.ndarray,
    prices: np.ndarray | None,
    batteries: list[Battery],
    controller,
    network: Network,
    dt_hours: float = 1.0,
    low_threshold: float = 30.0,
    high_threshold: float = 70.0,
    feeder_limit_kw: float = 60.0,
    rho_agents: float = 0.0,
    forecast_error_sigma_kw: float = 1.0,
    price_sigma: float = 0.0,
    pv_kw: np.ndarray | None = None,
    export_limit_kw: float | None = None,
    price_rule: str = "fixed",
    grid_price: float = 80.0,
    feed_in_price: float = 20.0,
) -> SimulationResult:
    """Run one day.

    Paper mode (defaults): pv_kw=None, price_rule="fixed".
        The feeder sees load minus battery only; prices are given in advance.

    Community mode: pass pv_kw, and optionally price_rule="sdr".
        The feeder sees load - PV - battery, so charging during PV surplus
        relieves the feeder instead of loading it.
        With "sdr" the price is set each step by community supply/demand,
        using the batteries' actions from the previous step (feedback).
    """
    n_steps, n_batteries = load_profiles_kw.shape

    if not 0.0 <= rho_agents <= 1.0:
        raise ValueError("rho_agents must be in [0, 1]")
    if price_rule not in ("fixed", "sdr"):
        raise ValueError("price_rule must be 'fixed' or 'sdr'")
    if price_rule == "fixed" and prices is None:
        raise ValueError("price_rule='fixed' needs prices")
    if price_rule == "sdr" and pv_kw is None:
        raise ValueError("price_rule='sdr' needs pv_kw")

    controllers = [controller] * n_batteries if callable(controller) else list(controller)
    pv = np.zeros(n_steps) if pv_kw is None else np.asarray(pv_kw, dtype=float)
    pv_per_house = pv / n_batteries
    export_limit = feeder_limit_kw if export_limit_kw is None else export_limit_kw

    # Correlated perception noise: sqrt(rho)*Z(t) + sqrt(1-rho)*eps_i(t).
    # Drawn in this order so the paper's random numbers are unchanged.
    Z = np.random.normal(0, 1, n_steps)
    eps = np.random.normal(0, 1, (n_steps, n_batteries))
    noise = (np.sqrt(rho_agents) * Z[:, None] + np.sqrt(1 - rho_agents) * eps)
    noise *= forecast_error_sigma_kw

    power = np.zeros((n_steps, n_batteries))
    feeder_net = np.zeros(n_steps)
    price_hist = np.zeros(n_steps)
    prev_actions = np.zeros(n_batteries)

    for t in range(n_steps):
        load_t = load_profiles_kw[t]

        # 1. Price for this step.
        if price_rule == "fixed":
            price_t = float(prices[t])
        else:
            house_net = load_t - pv_per_house[t] - prev_actions  # > 0 imports
            supply = float(np.sum(np.maximum(0.0, -house_net)))
            demand = float(np.sum(np.maximum(0.0, house_net)))
            price_t = sdr_price(supply, demand, grid_price, feed_in_price)
        price_hist[t] = price_t

        # 2. Each battery decides on its own noisy view of the price.
        perceived_feeder_kw = float(load_t.sum() - pv[t])
        actions = np.zeros(n_batteries)
        for i, battery in enumerate(batteries):
            nbrs = network.agents[i].neighbors
            neighbor_avg_kw = float(np.mean(prev_actions[nbrs])) if nbrs else 0.0
            requested_kw = controllers[i](
                price=price_t + float(noise[t, i]) * price_sigma,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                max_power_kw=battery.max_charge_kw,
                soc=battery.soc,
                feeder_load_kw=perceived_feeder_kw,
                feeder_limit_kw=feeder_limit_kw,
                neighbor_avg_kw=neighbor_avg_kw,
                agent_id=i,
                t=t,
            )
            actions[i] = battery.apply_power(requested_kw, dt_hours)

        # 3. Feeder flow after the batteries act.
        power[t] = actions
        prev_actions = actions
        feeder_net[t] = load_t.sum() - pv[t] - actions.sum()

    import_viol = feeder_net > feeder_limit_kw
    export_viol = feeder_net < -export_limit if pv_kw is not None else np.zeros(n_steps, bool)

    return SimulationResult(
        per_battery_power_kw=power,
        aggregate_battery_power_kw=power.sum(axis=1),
        feeder_net_kw=feeder_net,
        feeder_violations=import_viol | export_viol,
        import_violations=import_viol,
        export_violations=export_viol,
        price=price_hist,
    )
