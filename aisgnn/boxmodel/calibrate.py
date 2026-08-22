"""Calibration of the box-model process parameters against observations.

The geometry of each cavity is prescribed; what remains free is a small set of
*shared* process parameters -- the turbulent exchange velocity, the overturning
and export coefficients, the polynya heat-loss velocity and the outflow
entrainment fraction.  These are fitted once, on a small number of well-observed
cavities spanning the cold and warm regimes, and then held fixed everywhere
else.  Every other cavity is therefore a prediction of the model rather than a
fit, which is what makes the resulting bifurcation structure meaningful.

Observed melt fluxes are the steady-state estimates of Rignot et al. (2013),
cross-checked against the satellite-derived rates of Adusumilli et al. (2020).
Off-shelf CDW temperatures and polynya sea-ice production rates are
order-of-magnitude values from the regional literature; both are carried through
the sensitivity analysis rather than treated as exact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from .cavity import CAVITIES, BoxParams, CavityBoxModel

#: Parameters fitted during calibration; all are strictly positive and are
#: therefore optimised in log space.
FREE_PARAMETERS = ("c_ovt", "c_dsw", "c_exp", "lambda_atm", "eps_out")


@dataclass(frozen=True)
class Observation:
    """Observational constraints on one cavity's present-day state."""

    cavity: str
    melt_flux: float             # Gt/yr
    melt_flux_err: float         # Gt/yr, 1 sigma
    regime: str                  # 'cold' or 'warm'
    T_cdw: float                 # off-shelf deep-water temperature, degC
    sigma: float                 # polynya sea-ice production, m/yr
    T_shelf: float | None = None  # observed coastal-box temperature, degC
    S_shelf: float | None = None  # observed coastal-box salinity, g/kg
    q_overturn: float | None = None  # cavity ventilation flux, Sv


#: Present-day constraints.  Melt fluxes follow Rignot et al. (2013).
#:
#: The overturning constraints matter more than their nominal precision might
#: suggest.  Melt in this model is set by the competition between ventilation
#: (which supplies heat at the inflow temperature) and turbulent exchange at the
#: ice base (which draws the cavity towards the local freezing point).  Leaving
#: the ventilation free lets the optimiser run the overturning at several times
#: the observed value, which pins the cavity to the inflow temperature and
#: overestimates cold-cavity melt by an order of magnitude.
OBSERVATIONS: dict[str, Observation] = {
    o.cavity: o for o in (
        Observation("Filchner-Ronne", 155.4, 45.0, "cold", 0.5, 10.0, -1.90, 34.80, 1.6),
        Observation("Ross", 47.7, 34.0, "cold", 0.3, 12.0, -1.85, 34.75, 1.0),
        Observation("Amery", 35.5, 23.0, "cold", -0.3, 8.0),
        Observation("Fimbul", 26.8, 14.0, "cold", 0.0, 5.0),
        Observation("Larsen C", 20.7, 67.0, "cold", -0.5, 5.0),
        Observation("Riiser-Larsen", 9.4, 25.0, "cold", 0.0, 5.0),
        Observation("Shackleton", 72.6, 15.0, "warm", 0.3, 6.0),
        Observation("Totten", 63.2, 4.0, "warm", 0.5, 5.0),
        Observation("Getz", 144.9, 14.0, "warm", 0.9, 6.0, None, None, 0.5),
        Observation("Pine Island", 101.2, 8.0, "warm", 1.2, 4.0, -0.60, 34.35, 0.30),
        Observation("Thwaites", 97.5, 7.0, "warm", 1.1, 4.0),
    )
}

#: Cavities used to fit the shared parameters: two cold, two warm.
CALIBRATION_SET = ("Filchner-Ronne", "Ross", "Getz", "Pine Island")

#: Everything else is held out and predicted.
VALIDATION_SET = tuple(c for c in OBSERVATIONS if c not in CALIBRATION_SET)


@dataclass
class CalibrationResult:
    """Outcome of a parameter fit."""

    params: dict[str, float]
    cost: float
    success: bool
    message: str
    fitted: dict[str, dict[str, float]]     # cavity -> modelled diagnostics
    predicted: dict[str, dict[str, float]]  # held-out cavities

    def apply(self, base: BoxParams) -> BoxParams:
        """Return ``base`` with the calibrated process parameters substituted."""
        return replace(base, **self.params)


# --------------------------------------------------------------------------- #
# Forward evaluation
# --------------------------------------------------------------------------- #

def evaluate(cavity: str, params: dict[str, float],
             obs: Observation | None = None) -> dict[str, float]:
    """Solve for the attractor of one cavity and return its diagnostics.

    ``regime_ok`` reports whether the observed regime is actually an attractor
    for these parameters.  When it is not, the diagnostics of whichever
    attractor the cavity does settle on are returned anyway, so that the
    optimiser still sees a gradient instead of a flat penalty wall.
    """
    obs = obs or OBSERVATIONS[cavity]
    base = BoxParams(CAVITIES[cavity], T_cdw=obs.T_cdw, sigma=obs.sigma, **params)
    model = CavityBoxModel(base)

    regime_ok = True
    try:
        x = model.equilibrium(obs.regime)
    except RuntimeError:
        regime_ok = False
        try:
            x = model.equilibrium(obs.regime, require_stable=False)
        except RuntimeError:
            return {"melt_flux": np.nan, "melt_rate": np.nan, "T_shelf": np.nan,
                    "S_shelf": np.nan, "T_cavity": np.nan, "S_cavity": np.nan,
                    "chi": np.nan, "q_total": np.nan,
                    "stable": 0.0, "regime_ok": 0.0}

    d = model.diagnostics(x)
    stable, _ = model.is_stable(x)
    return {"melt_flux": d.melt_flux, "melt_rate": d.melt_rate,
            "T_shelf": float(x[2]), "S_shelf": float(x[3]),
            "T_cavity": float(x[0]), "S_cavity": float(x[1]),
            "chi": d.chi_dsw, "q_total": d.q_total,
            "stable": float(stable), "regime_ok": float(regime_ok)}


def _residuals(log_theta: np.ndarray, cavities: tuple[str, ...]) -> np.ndarray:
    """Weighted residuals for the least-squares fit."""
    params = {k: float(np.exp(v)) for k, v in zip(FREE_PARAMETERS, log_theta)}
    # eps_out is a fraction; keep it inside (0, 1].
    params["eps_out"] = min(params["eps_out"], 1.0)

    res: list[float] = []
    for cav in cavities:
        obs = OBSERVATIONS[cav]
        got = evaluate(cav, params, obs)

        if not np.isfinite(got["melt_flux"]) or got["melt_flux"] <= 0.0:
            res.extend([20.0, 0.0, 0.0, 0.0, 20.0])
            continue

        # Fit in log space: melt fluxes span two orders of magnitude across the
        # calibration set, so absolute residuals would be dominated by Getz.
        res.append(np.log(got["melt_flux"] / obs.melt_flux)
                   / np.log1p(obs.melt_flux_err / obs.melt_flux))

        res.append((got["T_shelf"] - obs.T_shelf) / 0.3 if obs.T_shelf is not None else 0.0)
        res.append((got["S_shelf"] - obs.S_shelf) / 0.1 if obs.S_shelf is not None else 0.0)

        # Ventilation, also in log space with a factor-of-two tolerance.
        if obs.q_overturn is not None and got["q_total"] > 0.0:
            res.append(np.log((got["q_total"] / 1e6) / obs.q_overturn) / np.log(2.0))
        else:
            res.append(0.0)

        # The observed regime must be a genuine attractor.  A parameter set that
        # reproduces the observed melt flux on an unstable branch has not
        # reproduced anything, so this term dominates when it fires.
        want_cold = obs.regime == "cold"
        chi_target = 1.0 if want_cold else 0.0
        res.append(8.0 * (got["chi"] - chi_target) if not got["regime_ok"] else 0.0)

    return np.asarray(res)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def calibrate(cavities: tuple[str, ...] = CALIBRATION_SET,
              start: dict[str, float] | None = None,
              bounds: dict[str, tuple[float, float]] | None = None,
              verbose: bool = False) -> CalibrationResult:
    """Fit the shared process parameters to the calibration cavities.

    Parameters
    ----------
    cavities
        Cavities whose observed melt flux (and, where available, coastal
        temperature and salinity) constrain the fit.
    start
        Initial guess; defaults to the values declared on :class:`BoxParams`.
    bounds
        Per-parameter ``(low, high)`` bounds in physical units.

    Returns
    -------
    CalibrationResult
        Fitted parameters plus modelled diagnostics for both the calibration
        cavities and the held-out validation cavities.
    """
    defaults = BoxParams(CAVITIES["Filchner-Ronne"])
    start = start or {k: getattr(defaults, k) for k in FREE_PARAMETERS}
    bounds = bounds or {
        "c_ovt": (1e-7, 1e-3),
        "c_dsw": (1e-5, 1.0),
        "c_exp": (1e-6, 1e-2),
        "lambda_atm": (1e-6, 1e-2),
        "eps_out": (0.01, 1.0),
    }

    x0 = np.array([np.log(start[k]) for k in FREE_PARAMETERS])
    lo = np.array([np.log(bounds[k][0]) for k in FREE_PARAMETERS])
    hi = np.array([np.log(bounds[k][1]) for k in FREE_PARAMETERS])

    sol = least_squares(_residuals, x0, args=(cavities,), bounds=(lo, hi),
                        method="trf", diff_step=1e-3, xtol=1e-10, ftol=1e-10,
                        verbose=2 if verbose else 0)

    params = {k: float(np.exp(v)) for k, v in zip(FREE_PARAMETERS, sol.x)}
    fitted = {c: evaluate(c, params) for c in cavities}
    predicted = {c: evaluate(c, params) for c in OBSERVATIONS if c not in cavities}

    return CalibrationResult(params=params, cost=float(sol.cost),
                             success=bool(sol.success), message=str(sol.message),
                             fitted=fitted, predicted=predicted)


def report(result: CalibrationResult) -> str:
    """Human-readable calibration summary."""
    lines = ["Calibrated process parameters:"]
    for k, v in result.params.items():
        lines.append(f"  {k:12s} = {v:.4g}")
    lines.append(f"  cost = {result.cost:.4g}   success = {result.success}")

    for title, block in (("Calibration cavities", result.fitted),
                         ("Held-out predictions", result.predicted)):
        lines.append("")
        lines.append(f"{title}:")
        lines.append(f"  {'cavity':16s} {'model':>9s} {'obs':>9s} {'ratio':>7s} "
                     f"{'regime':>7s} {'chi':>6s} {'q_Sv':>6s} {'ok':>4s}")
        for cav, got in block.items():
            obs = OBSERVATIONS[cav]
            ratio = got["melt_flux"] / obs.melt_flux if np.isfinite(got["melt_flux"]) else np.nan
            lines.append(f"  {cav:16s} {got['melt_flux']:9.1f} {obs.melt_flux:9.1f} "
                         f"{ratio:7.2f} {obs.regime:>7s} {got['chi']:6.2f} "
                         f"{got['q_total'] / 1e6:6.2f} "
                         f"{'yes' if got.get('regime_ok', 0.0) else 'NO':>4s}")

    return "\n".join(lines)
