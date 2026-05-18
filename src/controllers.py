from __future__ import annotations
from typing import List
import random

#-------------------------------------------------
# CONTROLLER IDEA:
# Controllers decide what the battery should do.
# Battery decides what CAN be actually possible.
#--------------------------------------------------

def tou_controller(
    price: float,
    low_threshold: float,
    high_threshold: float,
    max_power_kw: float,
    **_ignored,
) -> float:
    """
    Simple Time-of-Use controller.

    - Charge when price is low
    - Discharge when price is high
    - Otherwise do nothing

    Returns requested power (kW):
        negative = charge
        positive = discharge
    """

    if price <= low_threshold:
        return -max_power_kw  # charge

    if price >= high_threshold:
        return max_power_kw  # discharge

    return 0.0


# -------------------------------------------------
# RANDOMIZED CONTROLLER (for coordination test)
# -------------------------------------------------

def randomized_tou_controller(
    price: float,
    low_threshold: float,
    high_threshold: float,
    max_power_kw: float,
    randomness: float,
    **_ignored,
) -> float:
    """
    Same as TOU but introduces randomness to avoid synchronization.
    randomness: value in [0,1] controlling probability of acting.
    """

    if random.random() > randomness:
        return 0.0

    return tou_controller(
        price=price,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        max_power_kw=max_power_kw,
    )

# -------------------------------------------------
# FEEDER-AWARE CAP CONTROLLER
# -------------------------------------------------

def capped_controller(requested_kw: float, feeder_load_kw: float, feeder_limit_kw: float) -> float:
    """
    Caps battery action if feeder is close to limit.

    This is a VERY simple coordination rule.
    """

    if feeder_load_kw >= feeder_limit_kw:
        return 0.0

    return requested_kw

# -------------------------------------------------
# TOU + CAP COORDINATION
# -------------------------------------------------


def capped_tou_controller(
    price: float,
    low_threshold: float,
    high_threshold: float,
    max_power_kw: float,
    feeder_load_kw: float,
    feeder_limit_kw: float,
    **_ignored,
) -> float:
    """
    First request a standard TOU action, then cap it using feeder stress.
    """

    requested_kw = tou_controller(
        price=price,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        max_power_kw=max_power_kw,
    )

    return capped_controller(
        requested_kw=requested_kw,
        feeder_load_kw=feeder_load_kw,
        feeder_limit_kw=feeder_limit_kw,
    )

# -------------------------------------------------
# SOFT (PROPORTIONAL) CAP CONTROLLER
# -------------------------------------------------

def soft_capped_tou_controller(
    price: float,
    low_threshold: float,
    high_threshold: float,
    max_power_kw: float,
    feeder_load_kw: float,
    feeder_limit_kw: float,
    softness: float = 0.3,
    **_ignored,
) -> float:
    """
    Proportional (soft) coordination:

    - Start from TOU request
    - Gradually scale action down as feeder approaches the limit

    softness: fraction of the feeder limit that defines the transition band
              (e.g., 0.3 means start reducing at 70% of the limit)
    """

    # Base request from TOU
    requested_kw = tou_controller(
        price=price,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        max_power_kw=max_power_kw,
    )

    # Define a linear scaling region
    start_reduce = feeder_limit_kw * (1.0 - softness)

    if feeder_load_kw <= start_reduce:
        # Plenty of margin: allow full action
        return requested_kw

    if feeder_load_kw >= feeder_limit_kw:
        # At/above limit: block action that worsens the situation
        # (allow discharge if it helps reduce load, block charging)
        return max(requested_kw, 0.0)

    # Between start_reduce and feeder_limit: scale linearly from 1 → 0
    margin = feeder_limit_kw - feeder_load_kw
    band = feeder_limit_kw - start_reduce  # = feeder_limit_kw * softness
    scale = max(0.0, min(1.0, margin / band))

    return requested_kw * scale

# -------------------------------------------------
# AGGREGATION HELPER (optional)
# -------------------------------------------------

def aggregate_requests(requests: List[float]) -> float:
    """
    Sum of all battery requested powers.
    """

    return sum(requests)
