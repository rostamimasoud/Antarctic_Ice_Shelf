"""Reduced flowline ice model for the grounding-line response.

Integrates a marine grounding line under a prescribed melt history, to ask one
comparative question: how much more retreat follows a melt *regime shift* than
follows the same melt increase applied gradually.

This is a sensitivity experiment, not a projection, and two design choices are
deliberate and load-bearing:

* **The flux law is calibrated, not derived.** Grounding-line flux follows a
  power law ``q = k h^beta`` with ``beta = n + 2`` from Schoof (2007), but the
  coefficient ``k`` is set so that the chosen initial state is in mass balance.
  Using published Tsai et al. (2015) coefficients directly gives a flux three
  orders of magnitude below the accumulation input for a catchment of this size,
  and the grounding line then advances without limit -- a numerical artefact of
  mismatched scales rather than ice physics.
* **Melt acts through buttressing.** Basal melt thins the shelf, which reduces
  the back-stress on the grounded ice and raises the flux. That is the pathway
  by which ocean forcing reaches the grounding line, and it is represented by a
  single sensitivity coefficient that is varied rather than tuned.

On a retrograde bed (``bed_slope > 0`` here, since retreat is towards smaller
``x``) the model reproduces the marine ice sheet instability: retreat into
deeper water thickens the grounding line, which raises the flux, which drives
further retreat until the grounding line runs out of domain.  On a prograde bed
it instead settles at a new, further-inland equilibrium.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..config import CONST

N_GLEN = 3.0
FLUX_EXPONENT = N_GLEN + 2.0        # Schoof (2007) grounding-line flux scaling
OCEAN_AREA = 3.618e14               # m2, for sea-level equivalent


@dataclass
class FlowlineConfig:
    """Geometry, forcing and sensitivity of one flowline."""

    name: str = "idealised"
    width: float = 50.0e3           # catchment width, m
    accumulation: float = 0.3       # m ice per year
    bed_slope: float = -1.0e-3      # d(bed elevation)/dx.  Retreat is towards
    #                                 smaller x, so a POSITIVE slope means the
    #                                 bed deepens inland: that is the retrograde,
    #                                 unstable configuration.  The default is
    #                                 negative, i.e. prograde and stable.
    bed_depth_at_origin: float = -400.0
    x_init: float = 300.0e3         # initial grounding-line position, m
    melt_reference: float = 1.0     # melt the initial balance is defined at, m/yr
    melt_sensitivity: float = 0.6   # fractional flux increase per unit melt ratio
    x_min: float = 10.0e3           # inland limit of the domain

    def bed(self, x: np.ndarray | float) -> np.ndarray | float:
        """Bed elevation, m, negative below sea level."""
        return self.bed_depth_at_origin + self.bed_slope * np.asarray(x, float)

    @property
    def retrograde(self) -> bool:
        """True when the bed deepens inland, the marine-ice-sheet-unstable case.

        Retreat moves towards smaller ``x`` and ``bed = b0 + slope * x``, so the
        bed deepens inland exactly when ``slope > 0``.
        """
        return self.bed_slope > 0.0


def flotation_thickness(x: float, cfg: FlowlineConfig) -> float:
    """Ice thickness at flotation over the bed at ``x``."""
    return max(-cfg.bed(x) * CONST.rho_sw / CONST.rho_i, 1.0)


def calibrate_flux(cfg: FlowlineConfig) -> float:
    """Flux coefficient placing the initial grounding line in mass balance.

    Solves ``k h0^beta = a x0`` so that the control run with melt held at
    ``melt_reference`` neither advances nor retreats, and every subsequent
    change is attributable to the melt forcing rather than to a spurious initial
    imbalance.
    """
    h0 = flotation_thickness(cfg.x_init, cfg)
    return float(cfg.accumulation * cfg.x_init / h0 ** FLUX_EXPONENT)


def grounding_line_flux(thickness: float, melt: float, k: float,
                        cfg: FlowlineConfig) -> float:
    """Flux through the grounding line, m2/yr, including the melt-buttressing term."""
    buttress = 1.0 + cfg.melt_sensitivity * (melt - cfg.melt_reference) / max(
        cfg.melt_reference, 1e-6)
    return float(k * thickness ** FLUX_EXPONENT * max(buttress, 0.0))


@dataclass
class RetreatResult:
    """Grounding-line trajectory under a melt history."""

    time: np.ndarray                # years
    position: np.ndarray            # m
    thickness: np.ndarray           # m at the grounding line
    melt: np.ndarray                # m/yr
    sea_level: np.ndarray           # cumulative contribution, mm
    collapsed: bool
    time_of_collapse: float | None

    @property
    def total_retreat(self) -> float:
        return float(self.position[0] - self.position[-1])


def integrate_flowline(cfg: FlowlineConfig, melt_history: Callable[[float], float],
                       years: float = 500.0, dt: float = 0.25,
                       k: float | None = None) -> RetreatResult:
    """Integrate grounding-line position under a melt history.

    Mass conservation over the grounded domain gives

        ``dx_g/dt = (a x_g - q_g) / h_g``,

    the imbalance between accumulation over the catchment and discharge through
    the grounding line, divided by the thickness of the column being added or
    removed.
    """
    if k is None:
        k = calibrate_flux(cfg)

    n = int(years / dt) + 1
    t = np.linspace(0.0, years, n)

    x = cfg.x_init
    pos, thick, melts, sea_level = (np.empty(n) for _ in range(4))
    volume_lost = 0.0
    collapsed = False
    t_collapse = None

    for i, ti in enumerate(t):
        m = float(melt_history(ti))
        h = flotation_thickness(x, cfg)
        q = grounding_line_flux(h, m, k, cfg)

        dx = (cfg.accumulation * x - q) / h
        x = x + dx * dt

        if x <= cfg.x_min:
            x = cfg.x_min
            if not collapsed:
                collapsed = True
                t_collapse = float(ti)

        pos[i], thick[i], melts[i] = x, h, m
        if i > 0:
            volume_lost += max(pos[i - 1] - pos[i], 0.0) * cfg.width * h
        sea_level[i] = volume_lost * CONST.rho_i / CONST.rho_fw / OCEAN_AREA * 1000.0

    return RetreatResult(t, pos, thick, melts, sea_level, collapsed, t_collapse)


# --------------------------------------------------------------------------- #
# Melt histories
# --------------------------------------------------------------------------- #

def constant_melt(rate: float) -> Callable[[float], float]:
    return lambda t: rate


def gradual_melt(start: float, end: float, years: float) -> Callable[[float], float]:
    """Linear ramp over the full integration, holding at ``end`` afterwards."""
    def history(t: float) -> float:
        return start + (end - start) * min(max(t / years, 0.0), 1.0)
    return history


def regime_shift_melt(before: float, after: float, t_shift: float,
                      width: float = 5.0) -> Callable[[float], float]:
    """An abrupt jump between two melt levels, smoothed over ``width`` years.

    This is the melt history a cavity crossing a saddle-node produces.
    """
    def history(t: float) -> float:
        return before + (after - before) * 0.5 * (1.0 + np.tanh((t - t_shift) / width))
    return history


def compare_histories(cfg: FlowlineConfig, before: float, after: float,
                      t_shift: float = 100.0, years: float = 500.0,
                      **kwargs) -> dict[str, RetreatResult]:
    """Same endpoint melt reached abruptly, gradually, or not at all.

    All three share one calibration, so the control is exactly in balance and
    any retreat in the other two is caused by the melt change alone.
    """
    k = calibrate_flux(cfg)
    return {
        "control": integrate_flowline(cfg, constant_melt(before), years, k=k, **kwargs),
        "gradual": integrate_flowline(cfg, gradual_melt(before, after, years),
                                      years, k=k, **kwargs),
        "abrupt": integrate_flowline(cfg, regime_shift_melt(before, after, t_shift),
                                     years, k=k, **kwargs),
    }
