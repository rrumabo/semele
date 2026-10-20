"""Local price rules for an energy community.

Two rules:
    pv_shaped_price : fixed in advance from the PV shape (the paper's rule).
    sdr_price       : set every step by the community's own supply and demand.
"""
from __future__ import annotations

import numpy as np


def pv_shaped_price(pv_kw: np.ndarray, high: float = 80.0, swing: float = 60.0) -> np.ndarray:
    """Price falls as PV rises: high - swing * PV/PV_max. Same as the paper."""
    pv_norm = pv_kw / max(float(np.max(pv_kw)), 1e-9)
    return high - swing * pv_norm


def sdr_price(supply_kw: float, demand_kw: float, grid_price: float, feed_in_price: float) -> float:
    """Supply-demand-ratio (SDR) local buying price.

    supply_kw : total surplus offered by households that export.
    demand_kw : total shortfall of households that import.

    SDR = supply / demand.
        SDR = 0  -> nobody sells locally -> price = grid_price.
        SDR >= 1 -> local surplus covers everyone -> price = feed_in_price.
        between  -> smooth fall from grid_price to feed_in_price.

    The formula follows the SDR energy-sharing mechanism of Liu et al.,
    IEEE Trans. Power Systems, 2017.
    """
    if demand_kw <= 0.0:
        return feed_in_price
    sdr = supply_kw / demand_kw
    if sdr >= 1.0:
        return feed_in_price
    sell = (grid_price * feed_in_price) / ((grid_price - feed_in_price) * sdr + feed_in_price)
    return sell * sdr + grid_price * (1.0 - sdr)
