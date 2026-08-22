"""Rate-induced tipping experiments.

A system that is bistable can be driven out of its basin of attraction by a
forcing that changes *too quickly*, even when the forcing never crosses the
bifurcation threshold that would be needed to tip it under quasi-static change.
This module ramps a control parameter at a prescribed rate, detects whether the
cavity leaves the cold regime, and locates the critical ramp rate by bisection.

The distinction that matters for the results:

``B-tipping``
    the quasi-static threshold -- the saddle-node located by
    :mod:`aisgnn.boxmodel.continuation`.
``R-tipping``
    the largest parameter value the system tolerates when the forcing is ramped
    at a finite rate.  When this is smaller than the B-tipping threshold, the
    pace of change has done the tipping, not its magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cavity import BoxParams, CavityBoxModel

#: A cavity is deemed to have tipped once the DSW inflow fraction falls below
#: this value, i.e. the cavity is no longer ventilated by dense shelf water.
CHI_TIPPED = 0.5

#: Minimum years integrated after the ramp ends, regardless of ramp length.
#:
#: This is not a safety margin but a correctness requirement.  The cavity and
#: coastal boxes equilibrate over centuries to millennia -- the spin-ups used to
#: locate attractors run for 8000 years -- so relaxing for a small multiple of a
#: short ramp leaves the system mid-transient.  It then still looks
#: DSW-ventilated whatever the forcing has done, and every ramp is scored as
#: not tipped, producing a clean but entirely spurious "no rate-induced tipping"
#: result.
MIN_RELAX_YEARS = 5000.0


@dataclass
class RampResult:
    """Outcome of a single ramped-forcing integration."""

    parameter: str
    cavity: str
    p_start: float
    p_target: float
    tau_years: float             # ramp duration
    rate: float                  # (p_target - p_start) / tau, per year
    tipped: bool
    t_tip: float | None          # time of tipping, years
    p_tip: float | None          # parameter value at tipping
    t: np.ndarray                # output times, years
    x: np.ndarray                # state trajectory
    melt: np.ndarray             # melt rate, m/yr
    chi: np.ndarray              # DSW inflow fraction


@dataclass
class RTippingThreshold:
    """Critical ramp rate separating tipped from untipped outcomes."""

    parameter: str
    cavity: str
    p_start: float
    p_target: float
    tau_critical: float          # shortest ramp that does NOT tip, years
    tau_tipping: float           # longest ramp that DOES tip, years
    rate_critical: float         # corresponding rate, per year
    p_bifurcation: float | None  # quasi-static (B-tipping) threshold
    threshold_reduction: float | None  # fractional reduction vs B-tipping


# --------------------------------------------------------------------------- #
# Ramp forcing
# --------------------------------------------------------------------------- #

def linear_ramp(parameter: str, p_start: float, p_target: float, tau: float):
    """Return a forcing callable ramping ``parameter`` linearly over ``tau`` years.

    The parameter is held at ``p_target`` after the ramp completes, so that the
    integration can continue long enough to establish whether the system settles
    back onto the cold branch or commits to the warm one.
    """
    def forcing(t_years: float) -> dict:
        frac = min(max(t_years / tau, 0.0), 1.0)
        return {parameter: p_start + (p_target - p_start) * frac}

    return forcing


def run_ramp(base: BoxParams, parameter: str, p_start: float, p_target: float,
             tau_years: float, x0: np.ndarray | None = None,
             relax_years: float | None = None, n_out: int = 2000,
             noise: float = 0.0, seed: int | None = None) -> RampResult:
    """Integrate the model while ramping ``parameter`` from ``p_start`` to ``p_target``.

    Parameters
    ----------
    relax_years
        Time integrated *after* the ramp ends with the forcing held constant.
        Defaults to twice the ramp length but never less than
        :data:`MIN_RELAX_YEARS`, so the cavity has time to commit to a branch.
    """
    if relax_years is None:
        relax_years = max(2.0 * tau_years, MIN_RELAX_YEARS)

    model = CavityBoxModel(base.with_control(**{parameter: p_start}))
    if x0 is None:
        x0 = model.steady_state(model.initial_state("cold"))

    total = tau_years + relax_years
    forcing = linear_ramp(parameter, p_start, p_target, tau_years)
    t, X = model.integrate(x0, years=total, n_out=n_out, forcing=forcing,
                           noise=noise, seed=seed)

    melt = np.empty(t.size)
    chi = np.empty(t.size)
    for i, (ti, xi) in enumerate(zip(t, X)):
        m = CavityBoxModel(base.with_control(**forcing(ti)))
        d = m.diagnostics(xi)
        melt[i], chi[i] = d.melt_rate, d.chi_dsw

    below = np.where(chi < CHI_TIPPED)[0]
    tipped = below.size > 0 and bool(chi[-1] < CHI_TIPPED)
    i_tip = int(below[0]) if below.size else None

    return RampResult(
        parameter=parameter, cavity=base.geom.name,
        p_start=p_start, p_target=p_target, tau_years=tau_years,
        rate=(p_target - p_start) / tau_years,
        tipped=tipped,
        t_tip=float(t[i_tip]) if i_tip is not None else None,
        p_tip=float(forcing(t[i_tip])[parameter]) if i_tip is not None else None,
        t=t, x=X, melt=melt, chi=chi,
    )


# --------------------------------------------------------------------------- #
# Critical rate
# --------------------------------------------------------------------------- #

def critical_rate(base: BoxParams, parameter: str, p_start: float, p_target: float,
                  tau_bounds: tuple[float, float] = (5.0, 5000.0),
                  tol: float = 0.02, max_iter: int = 40,
                  p_bifurcation: float | None = None,
                  **ramp_kwargs) -> RTippingThreshold | None:
    """Bisect on the ramp duration to find the critical forcing rate.

    The search assumes tipping is monotone in the ramp duration: a fast ramp
    tips and a slow one does not.  Both ends of ``tau_bounds`` are checked
    first, and ``None`` is returned when the bracket does not contain a
    transition -- either every rate tips (``p_target`` is beyond the
    quasi-static threshold) or none does.

    Returns
    -------
    RTippingThreshold or None
    """
    tau_fast, tau_slow = float(tau_bounds[0]), float(tau_bounds[1])

    fast = run_ramp(base, parameter, p_start, p_target, tau_fast, **ramp_kwargs)
    slow = run_ramp(base, parameter, p_start, p_target, tau_slow, **ramp_kwargs)

    if not fast.tipped or slow.tipped:
        # No rate-induced transition inside the bracket.
        return None

    for _ in range(max_iter):
        if (tau_slow - tau_fast) / tau_slow < tol:
            break
        mid = np.sqrt(tau_fast * tau_slow)          # geometric bisection
        if run_ramp(base, parameter, p_start, p_target, mid, **ramp_kwargs).tipped:
            tau_fast = mid
        else:
            tau_slow = mid

    reduction = None
    if p_bifurcation is not None and p_bifurcation != p_start:
        reduction = float((p_bifurcation - p_target) / (p_bifurcation - p_start))

    return RTippingThreshold(
        parameter=parameter, cavity=base.geom.name,
        p_start=p_start, p_target=p_target,
        tau_critical=tau_slow, tau_tipping=tau_fast,
        rate_critical=(p_target - p_start) / tau_slow,
        p_bifurcation=p_bifurcation,
        threshold_reduction=reduction,
    )


def rate_threshold_curve(base: BoxParams, parameter: str, p_start: float,
                         p_targets: np.ndarray,
                         tau_bounds: tuple[float, float] = (5.0, 5000.0),
                         p_bifurcation: float | None = None,
                         **ramp_kwargs) -> dict[str, np.ndarray]:
    """Map the critical ramp duration as a function of the target parameter value.

    This traces the boundary in the (target, rate) plane that separates safe
    from tipping trajectories.  Targets beyond the quasi-static threshold tip at
    every rate and are reported as ``tau_critical = inf``; targets for which no
    rate tips are reported as ``nan``.
    """
    taus = np.full(p_targets.size, np.nan)
    rates = np.full(p_targets.size, np.nan)

    for i, p_t in enumerate(p_targets):
        if p_bifurcation is not None and p_t >= p_bifurcation:
            taus[i] = np.inf
            rates[i] = 0.0
            continue
        res = critical_rate(base, parameter, p_start, float(p_t),
                            tau_bounds=tau_bounds, p_bifurcation=p_bifurcation,
                            **ramp_kwargs)
        if res is not None:
            taus[i] = res.tau_critical
            rates[i] = res.rate_critical

    return {"p_target": np.asarray(p_targets, float),
            "tau_critical": taus,
            "rate_critical": rates}
