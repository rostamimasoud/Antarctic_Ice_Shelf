"""Numerical continuation of box-model steady states.

Provides pseudo-arclength continuation, which -- unlike naive parameter
stepping -- follows a solution branch around a fold and therefore resolves the
unstable middle branch that connects the cold and warm states.  Saddle-node
bifurcations are located by sign changes in the parameter component of the
tangent and then refined by solving the extended fold system

.. math::

    F(x, p) = 0, \\qquad F_x(x, p)\\, v = 0, \\qquad v^\\mathsf{T} v - 1 = 0,

whose solution is a point at which the Jacobian has an exact zero eigenvalue.

All continuation arithmetic is carried out in scaled coordinates.  This matters:
the raw state mixes temperatures of order 1 degC with salinities of order 34
g/kg, and control parameters range over tens of units, so an unscaled arclength
norm is dominated by salinity and by the parameter.  The corrector then steps
far enough to land on a different branch, silently producing a continuous-looking
curve that is really two branches spliced together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from .cavity import NDIM, BoxParams, CavityBoxModel

CONTROL_PARAMETERS = ("T_cdw", "sigma", "gamma_star", "S_cdw")

#: Characteristic variation of each state variable, used to scale the arclength.
STATE_SCALE = np.array([1.0, 0.25, 1.0, 0.25])   # (T_c, S_c, T_s, S_s)

#: A corrected step further than this multiple of the arclength step has jumped
#: to another branch and is rejected.
JUMP_TOLERANCE = 3.0


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass
class Branch:
    """A continued solution branch in one control parameter."""

    parameter: str
    cavity: str
    p: np.ndarray = field(default_factory=lambda: np.empty(0))          # (n,)
    x: np.ndarray = field(default_factory=lambda: np.empty((0, NDIM)))  # (n, 4)
    stable: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    melt: np.ndarray = field(default_factory=lambda: np.empty(0))       # m/yr
    melt_flux: np.ndarray = field(default_factory=lambda: np.empty(0))  # Gt/yr
    chi: np.ndarray = field(default_factory=lambda: np.empty(0))
    q_total: np.ndarray = field(default_factory=lambda: np.empty(0))
    eig_max: np.ndarray = field(default_factory=lambda: np.empty(0))    # max Re(lambda)

    def __len__(self) -> int:
        return int(self.p.size)

    def segment(self, stable: bool) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(p, melt)`` restricted to the stable or unstable portion."""
        m = self.stable if stable else ~self.stable
        return self.p[m], self.melt[m]


@dataclass
class Fold:
    """A saddle-node bifurcation."""

    parameter: str
    cavity: str
    p: float                     # control-parameter value at the fold
    x: np.ndarray                # state at the fold
    melt: float                  # melt rate at the fold, m/yr
    direction: str               # 'cold_to_warm' or 'warm_to_cold'
    residual: float              # ||F|| at the refined fold
    refined: bool                # whether Newton refinement converged


@dataclass
class Hysteresis:
    """A bistable window bounded by two folds."""

    parameter: str
    cavity: str
    p_forward: float             # fold crossed when the parameter increases
    p_reverse: float             # fold crossed when it decreases
    width: float                 # |p_forward - p_reverse|
    melt_jump: float             # melt-rate discontinuity at the forward fold
    folds: tuple[Fold, ...]


# --------------------------------------------------------------------------- #
# The scaled steady-state system
# --------------------------------------------------------------------------- #

class _System:
    """``F(x, p) = 0`` for one control parameter, in scaled coordinates.

    Scaled variables are ``y = x / STATE_SCALE`` and ``w = p / p_scale``; the
    residual is additionally divided by the box volumes so that the cavity and
    coastal balances contribute comparably.
    """

    def __init__(self, base: BoxParams, parameter: str, p_scale: float = 1.0):
        if parameter not in CONTROL_PARAMETERS:
            raise ValueError(f"parameter must be one of {CONTROL_PARAMETERS}, got {parameter!r}")
        self.base = base
        self.parameter = parameter
        self.p_scale = float(p_scale) if p_scale else 1.0

        m = CavityBoxModel(base)
        s = np.array([1.0 / m.g.cavity_volume] * 2 + [1.0 / m.g.shelf_volume] * 2)
        self.res_scale = s / s.max()

    # -- coordinate maps ---------------------------------------------------- #

    def to_y(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, float) / STATE_SCALE

    def to_x(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, float) * STATE_SCALE

    def to_w(self, p: float) -> float:
        return float(p) / self.p_scale

    def to_p(self, w: float) -> float:
        return float(w) * self.p_scale

    # -- model -------------------------------------------------------------- #

    def model_at(self, p: float) -> CavityBoxModel:
        return CavityBoxModel(self.base.with_control(**{self.parameter: float(p)}))

    def F(self, y: np.ndarray, w: float) -> np.ndarray:
        """Scaled residual."""
        return self.model_at(self.to_p(w)).rhs(0.0, self.to_x(y)) / self.res_scale

    def F_y(self, y: np.ndarray, w: float, eps: float = 1e-7) -> np.ndarray:
        J = np.empty((NDIM, NDIM))
        for j in range(NDIM):
            h = eps * max(abs(y[j]), 1.0)
            yp, ym = y.copy(), y.copy()
            yp[j] += h
            ym[j] -= h
            J[:, j] = (self.F(yp, w) - self.F(ym, w)) / (2.0 * h)
        return J

    def F_w(self, y: np.ndarray, w: float, eps: float = 1e-6) -> np.ndarray:
        h = eps * max(abs(w), 1.0)
        return (self.F(y, w + h) - self.F(y, w - h)) / (2.0 * h)

    def solve(self, y0: np.ndarray, w: float) -> np.ndarray:
        sol = root(lambda u: self.F(u, w), y0, method="hybr", tol=1e-12)
        if not sol.success:
            raise RuntimeError(
                f"steady state failed at {self.parameter}={self.to_p(w):g}: {sol.message}")
        return sol.x


# --------------------------------------------------------------------------- #
# Pseudo-arclength continuation
# --------------------------------------------------------------------------- #

def _tangent(sys: _System, y: np.ndarray, w: float,
             prev: np.ndarray | None) -> np.ndarray:
    """Unit tangent to the solution curve, oriented consistently with ``prev``."""
    A = np.zeros((NDIM + 1, NDIM + 1))
    A[:NDIM, :NDIM] = sys.F_y(y, w)
    A[:NDIM, NDIM] = sys.F_w(y, w)
    A[NDIM, :] = prev if prev is not None else np.eye(NDIM + 1)[NDIM]

    rhs = np.zeros(NDIM + 1)
    rhs[NDIM] = 1.0
    try:
        tau = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        tau = np.linalg.lstsq(A, rhs, rcond=None)[0]

    norm = np.linalg.norm(tau)
    if norm == 0.0 or not np.isfinite(norm):
        raise RuntimeError("degenerate tangent")
    tau = tau / norm
    if prev is not None and float(tau @ prev) < 0.0:
        tau = -tau
    return tau


def _corrector(sys: _System, z_pred: np.ndarray, z_prev: np.ndarray,
               tau: np.ndarray, ds: float, max_iter: int = 20,
               tol: float = 1e-10) -> np.ndarray | None:
    """Newton correction back onto the curve within the arclength hyperplane."""
    z = z_pred.copy()
    G = np.full(NDIM + 1, np.inf)

    for _ in range(max_iter):
        y, w = z[:NDIM], float(z[NDIM])
        G = np.empty(NDIM + 1)
        G[:NDIM] = sys.F(y, w)
        G[NDIM] = float(tau @ (z - z_prev)) - ds

        if np.linalg.norm(G) < tol:
            return z

        J = np.zeros((NDIM + 1, NDIM + 1))
        J[:NDIM, :NDIM] = sys.F_y(y, w)
        J[:NDIM, NDIM] = sys.F_w(y, w)
        J[NDIM, :] = tau
        try:
            dz = np.linalg.solve(J, G)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(dz)):
            return None
        z = z - dz

    return z if np.linalg.norm(G) < 1e-7 else None


def continue_branch(base: BoxParams, parameter: str, x0: np.ndarray, p0: float,
                    p_min: float, p_max: float, ds: float = 0.02,
                    ds_min: float = 1e-7, ds_max: float = 0.05,
                    max_steps: int = 20000, direction: int = 1,
                    p_scale: float | None = None) -> Branch:
    """Trace a solution branch by pseudo-arclength continuation.

    Parameters
    ----------
    base
        Baseline parameters; ``parameter`` is varied from these.
    parameter
        Control parameter name, one of :data:`CONTROL_PARAMETERS`.
    x0, p0
        A converged starting point on the branch, in physical units.
    p_min, p_max
        Continuation stops once the parameter leaves this interval.
    ds
        Initial arclength step in scaled units; adapted within
        ``[ds_min, ds_max]``.
    direction
        ``+1`` to set off towards increasing ``parameter``, ``-1`` for decreasing.
    p_scale
        Characteristic parameter variation; defaults to one twentieth of the
        continuation interval, which keeps the tangent well conditioned.

    Returns
    -------
    Branch
        The traced branch, including any unstable segment between folds.
    """
    if p_scale is None:
        p_scale = max(abs(p_max - p_min) / 20.0, 1e-6)

    sys = _System(base, parameter, p_scale=p_scale)
    w0 = sys.to_w(p0)
    y = sys.solve(sys.to_y(x0), w0)
    z = np.concatenate([y, [w0]])

    tau = _tangent(sys, y, w0, None)
    if np.sign(tau[NDIM]) != np.sign(direction):
        tau = -tau

    ws, ys = [w0], [y.copy()]
    step = ds

    for _ in range(max_steps):
        z_new = _corrector(sys, z + step * tau, z, tau, step)

        # Reject a converged step that landed too far from the predictor: the
        # corrector has found a genuine solution, but on a different branch.
        # Without this the continuation silently splices the cold and warm
        # branches together and reports no unstable segment at all.
        if z_new is not None and np.linalg.norm(z_new - z) > JUMP_TOLERANCE * step:
            z_new = None

        if z_new is None:
            step *= 0.5
            if step < ds_min:
                break
            continue

        y_new, w_new = z_new[:NDIM], float(z_new[NDIM])
        try:
            tau = _tangent(sys, y_new, w_new, tau)
        except RuntimeError:
            break
        z = z_new

        ws.append(w_new)
        ys.append(y_new.copy())

        if not (p_min <= sys.to_p(w_new) <= p_max):
            break

        step = min(step * 1.1, ds_max)

    ps = np.array([sys.to_p(w) for w in ws])
    xs = np.array([sys.to_x(y) for y in ys])
    return _finalise(sys, parameter, base.geom.name, ps, xs)


def _finalise(sys: _System, parameter: str, cavity: str,
              ps: np.ndarray, xs: np.ndarray) -> Branch:
    """Attach stability and physical diagnostics to a traced branch."""
    n = ps.size
    stable = np.empty(n, bool)
    eig_max = np.empty(n)
    melt = np.empty(n)
    flux = np.empty(n)
    chi = np.empty(n)
    q = np.empty(n)

    for i in range(n):
        m = sys.model_at(ps[i])
        ev = np.linalg.eigvals(m.jacobian(xs[i]))
        eig_max[i] = ev.real.max()
        stable[i] = eig_max[i] < 0.0
        d = m.diagnostics(xs[i])
        melt[i], flux[i], chi[i], q[i] = d.melt_rate, d.melt_flux, d.chi_dsw, d.q_total

    return Branch(parameter=parameter, cavity=cavity, p=ps, x=xs, stable=stable,
                  melt=melt, melt_flux=flux, chi=chi, q_total=q, eig_max=eig_max)


# --------------------------------------------------------------------------- #
# Fold detection and refinement
# --------------------------------------------------------------------------- #

def _refine_fold(sys: _System, y0: np.ndarray, w0: float
                 ) -> tuple[np.ndarray, float, float, bool]:
    """Newton-refine a fold via the extended system ``F=0, F_y v=0, |v|=1``.

    The inner Jacobian uses a deliberately coarse finite-difference step: the
    outer solver differences ``G`` numerically, so a tight inner step would mean
    differencing a difference and the resulting second-derivative noise prevents
    convergence entirely.
    """
    inner_eps = 1e-5
    J = sys.F_y(y0, w0, eps=inner_eps)
    evals, evecs = np.linalg.eig(J)
    v0 = np.real(evecs[:, int(np.argmin(np.abs(evals)))])
    nv = np.linalg.norm(v0)
    v0 = v0 / nv if nv > 0 else np.ones(NDIM) / np.sqrt(NDIM)

    def G(u):
        y, v, w = u[:NDIM], u[NDIM:2 * NDIM], float(u[2 * NDIM])
        return np.concatenate([sys.F(y, w),
                               sys.F_y(y, w, eps=inner_eps) @ v,
                               [v @ v - 1.0]])

    sol = root(G, np.concatenate([y0, v0, [w0]]), method="hybr",
               tol=1e-10, options={"eps": 1e-4})
    if not sol.success:
        return y0, w0, float(np.linalg.norm(sys.F(y0, w0))), False

    y = sol.x[:NDIM]
    w = float(sol.x[2 * NDIM])
    return y, w, float(np.linalg.norm(sys.F(y, w))), True


def find_folds(base: BoxParams, branch: Branch,
               p_scale: float | None = None) -> list[Fold]:
    """Locate saddle-node bifurcations along a continued branch.

    Folds are detected where the branch reverses direction in the control
    parameter, which is also where the leading eigenvalue crosses zero.
    """
    if len(branch) < 3:
        return []
    if p_scale is None:
        span = float(branch.p.max() - branch.p.min())
        p_scale = max(span / 20.0, 1e-6)

    sys = _System(base, branch.parameter, p_scale=p_scale)
    folds: list[Fold] = []

    dp = np.diff(branch.p)
    turns = np.where(np.sign(dp[:-1]) * np.sign(dp[1:]) < 0)[0] + 1

    for i in turns:
        y, w, res, ok = _refine_fold(sys, sys.to_y(branch.x[i]), sys.to_w(branch.p[i]))
        p = sys.to_p(w)
        x = sys.to_x(y)
        melt = float(sys.model_at(p).diagnostics(x).melt_rate)

        # A fold at the top of the low-melt branch is the cold->warm transition;
        # one at the bottom of the high-melt branch is warm->cold.
        lo, hi = max(i - 5, 0), min(i + 6, len(branch))
        window = branch.melt[lo:hi]
        mid = 0.5 * (float(window.max()) + float(window.min()))
        direction = "cold_to_warm" if melt < mid else "warm_to_cold"

        folds.append(Fold(parameter=branch.parameter, cavity=branch.cavity, p=p, x=x,
                          melt=melt, direction=direction, residual=res, refined=ok))

    return folds


def hysteresis(base: BoxParams, branch: Branch) -> Hysteresis | None:
    """Quantify the bistable window enclosed by a pair of folds.

    Returns ``None`` when fewer than two folds are present, i.e. the branch is
    monostable over the interval that was continued.
    """
    folds = find_folds(base, branch)
    if len(folds) < 2:
        return None

    ordered = sorted(folds, key=lambda f: f.p)
    lo, hi = ordered[0], ordered[-1]

    inside = (branch.p > lo.p) & (branch.p < hi.p)
    jump = (float(branch.melt[inside].max() - branch.melt[inside].min())
            if inside.any() else abs(hi.melt - lo.melt))

    return Hysteresis(parameter=branch.parameter, cavity=branch.cavity,
                      p_forward=hi.p, p_reverse=lo.p, width=abs(hi.p - lo.p),
                      melt_jump=jump, folds=tuple(ordered))


# --------------------------------------------------------------------------- #
# High-level driver
# --------------------------------------------------------------------------- #

def bifurcation_diagram(base: BoxParams, parameter: str, p_start: float,
                        p_min: float, p_max: float, regime: str = "cold",
                        ds: float = 0.02, max_steps: int = 20000
                        ) -> tuple[Branch, list[Fold], Hysteresis | None]:
    """Continue a branch in both directions from ``p_start`` and analyse it.

    The branch is traced towards decreasing and then increasing ``parameter``
    and the halves are joined, so a single :class:`Branch` spans the whole
    S-shaped curve including the unstable middle segment.

    The starting point comes from :meth:`CavityBoxModel.equilibrium`, not from a
    bare Newton solve: the nominal initial guess lies outside the basin for
    several cavities, and Newton either fails outright or lands on the unstable
    branch, which starts the continuation in the wrong place.
    """
    model = CavityBoxModel(base.with_control(**{parameter: p_start}))
    x0 = model.equilibrium(regime)
    p_scale = max(abs(p_max - p_min) / 20.0, 1e-6)

    back = continue_branch(base, parameter, x0, p_start, p_min, p_max, ds=ds,
                           max_steps=max_steps, direction=-1, p_scale=p_scale)
    fwd = continue_branch(base, parameter, x0, p_start, p_min, p_max, ds=ds,
                          max_steps=max_steps, direction=+1, p_scale=p_scale)

    merged = Branch(
        parameter=parameter, cavity=base.geom.name,
        p=np.concatenate([back.p[::-1], fwd.p[1:]]),
        x=np.concatenate([back.x[::-1], fwd.x[1:]]),
        stable=np.concatenate([back.stable[::-1], fwd.stable[1:]]),
        melt=np.concatenate([back.melt[::-1], fwd.melt[1:]]),
        melt_flux=np.concatenate([back.melt_flux[::-1], fwd.melt_flux[1:]]),
        chi=np.concatenate([back.chi[::-1], fwd.chi[1:]]),
        q_total=np.concatenate([back.q_total[::-1], fwd.q_total[1:]]),
        eig_max=np.concatenate([back.eig_max[::-1], fwd.eig_max[1:]]),
    )

    return merged, find_folds(base, merged, p_scale), hysteresis(base, merged)


# --------------------------------------------------------------------------- #
# Two-parameter bistability map
# --------------------------------------------------------------------------- #

def bistability_map(base: BoxParams, x_parameter: str, x_values: np.ndarray,
                    y_parameter: str, y_values: np.ndarray,
                    regimes: tuple[str, str] = ("cold", "warm")
                    ) -> dict[str, np.ndarray]:
    """Map which regimes are attractors over a plane of two control parameters.

    For every point on the grid the model is relaxed from both a cold and a warm
    initial condition; a point is bistable when both relax to their own regime.
    This is the two-parameter view of the fold structure and is the natural way
    to present the results when only one of the two parameters actually carries
    a saddle-node.

    Returns
    -------
    dict
        ``cold``/``warm`` boolean grids of shape ``(len(y_values), len(x_values))``,
        ``bistable`` where both hold, and ``melt_cold``/``melt_warm`` grids with
        NaN where that regime is not an attractor.
    """
    nx, ny = len(x_values), len(y_values)
    cold = np.zeros((ny, nx), bool)
    warm = np.zeros((ny, nx), bool)
    melt_cold = np.full((ny, nx), np.nan)
    melt_warm = np.full((ny, nx), np.nan)

    for j, yv in enumerate(y_values):
        for i, xv in enumerate(x_values):
            p = base.with_control(**{x_parameter: float(xv), y_parameter: float(yv)})
            model = CavityBoxModel(p)
            for regime, mask, melt in ((regimes[0], cold, melt_cold),
                                       (regimes[1], warm, melt_warm)):
                try:
                    x = model.equilibrium(regime)
                except RuntimeError:
                    continue
                mask[j, i] = True
                melt[j, i] = model.diagnostics(x).melt_rate

    return {"x_parameter": x_parameter, "y_parameter": y_parameter,
            "x": np.asarray(x_values, float), "y": np.asarray(y_values, float),
            "cold": cold, "warm": warm, "bistable": cold & warm,
            "melt_cold": melt_cold, "melt_warm": melt_warm}
