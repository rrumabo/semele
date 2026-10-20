"""Battery controllers.

Every controller returns requested power in kW: < 0 charge, > 0 discharge.
Extra keyword arguments from the simulator are ignored (**_ignored).
"""
from __future__ import annotations


def tou_controller(price: float, low_threshold: float, high_threshold: float,
                   max_power_kw: float, **_ignored) -> float:
    """Charge at full power when price is low, discharge when high, else idle."""
    if price <= low_threshold:
        return -max_power_kw
    if price >= high_threshold:
        return max_power_kw
    return 0.0


def belief_neighborhood_controller(
    price: float,
    low_threshold: float,
    high_threshold: float,
    max_power_kw: float,
    feeder_load_kw: float,
    feeder_limit_kw: float,
    neighbor_avg_kw: float = 0.0,
    belief_weight: float = 0.5,
    **_ignored,
) -> float:
    """The paper's hybrid controller.

    Start from the TOU request. Only charging is scaled down, by
        scale = alpha * feeder_scale + (1 - alpha) * neighbor_scale
    with alpha = belief_weight.
        feeder_scale   : room left on the feeder (1 = empty, 0 = at limit).
        neighbor_scale : 1 if neighbours idled last step, 0 if they charged flat out.
    The neighbour term is the only place network topology enters.
    """
    request = tou_controller(price, low_threshold, high_threshold, max_power_kw)
    if request >= 0.0:
        return request

    alpha = max(0.0, min(1.0, belief_weight))
    feeder_scale = max(0.0, min(1.0, 1.0 - feeder_load_kw / feeder_limit_kw))
    neighbor_pressure = max(0.0, -neighbor_avg_kw) / max_power_kw
    neighbor_scale = max(0.0, min(1.0, 1.0 - neighbor_pressure))

    return request * (alpha * feeder_scale + (1.0 - alpha) * neighbor_scale)
