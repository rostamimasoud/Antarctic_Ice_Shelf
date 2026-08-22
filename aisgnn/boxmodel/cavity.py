"""Low-dimensional box model of ice-shelf cavity circulation.

The model resolves the coupled cavity / coastal-polynya system with four
prognostic variables and reproduces the two circulation regimes that make
ice-shelf cavities susceptible to irreversible transitions:

* a **cold, convective regime** in which brine rejection in the coastal polynya
  produces dense shelf water (DSW) that is dense enough to sink into the cavity,
  flushing it with water close to the surface freezing point.  Basal melt is low.
* a **warm, diffusive regime** in which the cavity is ventilated instead by
  modified warm deep water (mWDW).  Basal melt is high.

The two regimes are connected by a positive feedback that operates through the
salinity of the coastal box.  Meltwater leaving the cavity freshens the shelf,
lowering its density and suppressing DSW formation, which keeps the cavity warm
and melt high.  Conversely, low melt leaves brine rejection unopposed, DSW forms,
and the cavity stays cold.  For intermediate sea-ice formation rates both states
are stable, and the system possesses a pair of saddle-node bifurcations enclosing
a hysteresis loop.

State vector
------------
``x = (T_c, S_c, T_s, S_s)`` -- cavity temperature and salinity, coastal-box
temperature and salinity.

Control parameters
------------------
``T_cdw``       temperature of the Circumpolar Deep Water reservoir (degC)
``sigma``       sea-ice formation rate in the polynya (m of ice per year)
``gamma_star``  vertical exchange velocity at the ice-shelf front (m/s)

Lineage: the box structure follows Olbers and Hellmer (2010) and the PICO model
of Reese et al. (2018); the coastal polynya box and the resulting bistability
follow Saddier et al. (2026).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ..config import CONST

SEC_PER_YEAR = CONST.sec_per_year

STATE_NAMES = ("T_c", "S_c", "T_s", "S_s")
NDIM = len(STATE_NAMES)


# --------------------------------------------------------------------------- #
# Thermodynamic helpers
# --------------------------------------------------------------------------- #

def freezing_point(S: np.ndarray | float, z: float) -> np.ndarray | float:
    """Linearised in-situ freezing temperature (degC) at depth ``z`` (m, negative down)."""
    return CONST.lam1 * S + CONST.lam2 + CONST.lam3 * z


def density(T: np.ndarray | float, S: np.ndarray | float) -> np.ndarray | float:
    """Linear equation of state, potential density anomaly referenced to ``rho_sw``."""
    return CONST.rho_sw * (1.0 - CONST.alpha_T * (T - CONST.T0)
                           + CONST.beta_S * (S - CONST.S0))


def latent_heat_effective(T_f: float, T_ice: float) -> float:
    """Latent heat plus the sensible heat needed to warm ice to the freezing point."""
    return CONST.L_f + CONST.c_pi * (T_f - T_ice)


def smooth_step(x: np.ndarray | float, width: float) -> np.ndarray | float:
    """Logistic step, ``->0`` for ``x << 0`` and ``->1`` for ``x >> 0``.

    Used instead of a hard switch so that the steady-state problem stays
    differentiable and pseudo-arclength continuation can be applied.
    """
    return 0.5 * (1.0 + np.tanh(x / width))


def softplus(x: np.ndarray | float, width: float) -> np.ndarray | float:
    """Smooth approximation to ``max(x, 0)``, matching ``x`` for ``x >> width``.

    ``np.logaddexp`` is stable at both extremes, so no clipping is applied.
    Clipping the argument here would silently cap the function at
    ``width * clip`` and saturate the overturning fluxes that use it.
    """
    return width * np.logaddexp(0.0, np.asarray(x, dtype=float) / width)


# --------------------------------------------------------------------------- #
# Geometry and parameters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CavityGeometry:
    """Bulk geometry of one ice-shelf cavity and its adjacent coastal box.

    Values are order-of-magnitude estimates compiled from published ice-shelf
    areas, mean draughts and polynya extents; they set the timescales of the
    model rather than any tuned result, and every conclusion is checked for
    robustness by varying them (see :mod:`aisgnn.boxmodel.continuation`).
    """

    name: str
    area_ice: float          # ice-shelf basal area, m2
    draft: float             # mean ice draft, m (negative down)
    water_column: float      # mean cavity water-column thickness, m
    front_area: float        # cross-sectional area of the ice-shelf front, m2
    polynya_area: float      # coastal polynya area, m2
    shelf_area: float        # continental-shelf (coastal box) area, m2
    shelf_depth: float       # coastal box thickness, m
    aspect_ratio: float      # cavity length / width, dimensionless
    channel_width: float     # width of the dominant inflow channel, m

    @property
    def cavity_volume(self) -> float:
        return self.area_ice * self.water_column

    @property
    def shelf_volume(self) -> float:
        return self.shelf_area * self.shelf_depth


@dataclass(frozen=True)
class BoxParams:
    """Physical parameters and control parameters of the box model."""

    geom: CavityGeometry

    # --- control parameters ------------------------------------------------ #
    T_cdw: float = 0.5              # CDW temperature, degC
    S_cdw: float = 34.7             # CDW salinity, g/kg
    sigma: float = 3.0              # sea-ice formation rate, m ice per year
    gamma_star: float = 1.0e-4      # front vertical exchange velocity, m/s

    # --- process parameters ------------------------------------------------ #
    gamma_T: float = 5.0e-5         # effective turbulent heat exchange velocity, m/s.
    #                                 Held fixed rather than fitted: above about
    #                                 1e-4 melt becomes limited by ventilation
    #                                 rather than by exchange, so the calibration
    #                                 cost is flat in this direction and the
    #                                 optimiser walks to whatever bound it is
    #                                 given.  Larger than the local three-equation
    #                                 value because one box must stand in for the
    #                                 area-integrated exchange across a cavity
    #                                 with strong spatial structure.
    c_ovt: float = 1.0e-5           # meltwater-pump coefficient, m4 kg-1 s-1
    c_dsw: float = 5.0e-3           # dense-water gravity-current coefficient,
    #                                 m4 kg-1 s-1; multiplies the *front* area
    #                                 because the inflow is a density current
    #                                 through the ice-shelf front, not a
    #                                 buoyancy flux distributed over the base
    c_exp: float = 1.0e-4           # DSW export coefficient, m4 kg-1 s-1
    eps_out: float = 0.15           # fraction of cavity outflow entrained into
    #                                 the coastal box; the buoyant remainder
    #                                 leaves along the front without joining the
    #                                 dense-water formation region
    q_shelf: float = 5.0e4          # coastal box <-> open ocean exchange, m3/s
    lambda_atm: float = 5.0e-5      # surface heat-loss piston velocity, m/s
    T_ice: float = -20.0            # ice-column temperature, degC
    S_ice: float = 7.0              # sea-ice bulk salinity, g/kg
    T_surface: float = -1.9         # surface (polynya) temperature, degC

    # --- smoothing widths -------------------------------------------------- #
    drho_switch: float = 0.05       # density scale of the DSW/mWDW switch, kg/m3
    drho_soft: float = 5.0e-3       # softening scale of the overturning rectifier

    def with_control(self, **kwargs: float) -> "BoxParams":
        """Return a copy with one or more control parameters replaced."""
        return replace(self, **kwargs)

    @property
    def overturning_coefficient(self) -> float:
        """Meltwater-pump strength (m6 kg-1 s-1), scaled with the ice-base area."""
        return self.c_ovt * self.geom.area_ice

    @property
    def gravity_current_coefficient(self) -> float:
        """Dense-water inflow strength (m6 kg-1 s-1), scaled with the front area."""
        return self.c_dsw * self.geom.front_area

    @property
    def export_coefficient(self) -> float:
        """Dense-shelf-water export strength (m6 kg-1 s-1), scaled with polynya size."""
        return self.c_exp * self.geom.polynya_area

    @property
    def sigma_si(self) -> float:
        """Sea-ice formation rate converted to m/s."""
        return self.sigma / SEC_PER_YEAR


# --------------------------------------------------------------------------- #
# Cavity registry
# --------------------------------------------------------------------------- #

#: Bulk geometry for the cavities analysed in this study.
#:
#: ``shelf_area`` is the dense-water formation region adjacent to the cavity --
#: the part of the continental shelf whose properties are actually set by the
#: coastal polynya -- rather than the full shelf, which would dilute both the
#: brine flux and the surface heat loss by an order of magnitude.
CAVITIES: dict[str, CavityGeometry] = {
    g.name: g for g in (
        CavityGeometry("Filchner-Ronne", 4.30e11, -500.0, 400.0, 4.0e8, 3.0e10,
                       1.50e11, 500.0, 1.2, 1.5e5),
        CavityGeometry("Ross", 5.00e11, -400.0, 350.0, 4.5e8, 5.0e10,
                       2.50e11, 500.0, 1.1, 2.0e5),
        CavityGeometry("Amery", 6.00e10, -500.0, 500.0, 1.0e8, 8.0e9,
                       4.00e10, 500.0, 3.0, 4.0e4),
        CavityGeometry("Fimbul", 4.10e10, -300.0, 300.0, 6.0e7, 4.0e9,
                       2.00e10, 400.0, 0.6, 3.0e4),
        CavityGeometry("Larsen C", 4.60e10, -300.0, 350.0, 6.0e7, 5.0e9,
                       2.50e10, 400.0, 1.4, 4.0e4),
        CavityGeometry("Riiser-Larsen", 4.30e10, -250.0, 300.0, 5.0e7, 4.0e9,
                       2.00e10, 400.0, 1.3, 3.0e4),
        CavityGeometry("Shackleton", 3.30e10, -300.0, 350.0, 5.0e7, 6.0e9,
                       3.00e10, 400.0, 2.2, 3.5e4),
        CavityGeometry("Totten", 6.40e9, -600.0, 500.0, 2.0e7, 2.0e9,
                       1.00e10, 500.0, 2.5, 1.5e4),
        CavityGeometry("Getz", 3.40e10, -300.0, 400.0, 8.0e7, 5.0e9,
                       2.50e10, 500.0, 0.4, 5.0e4),
        CavityGeometry("Pine Island", 6.00e9, -600.0, 500.0, 1.5e7, 1.0e9,
                       5.00e9, 600.0, 3.5, 1.2e4),
        CavityGeometry("Thwaites", 5.00e9, -500.0, 450.0, 2.0e7, 1.5e9,
                       7.50e9, 600.0, 1.8, 2.0e4),
    )
}

#: Nine cavities spanning cold, intermediate and warm regimes.
NINE_CAVITIES = ("Filchner-Ronne", "Ross", "Amery", "Fimbul", "Larsen C",
                 "Riiser-Larsen", "Shackleton", "Totten", "Getz")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

@dataclass
class Diagnostics:
    """Derived quantities at a given state, in physical units."""

    melt_rate: float          # m of ice per year, area-averaged
    melt_flux: float          # Gt per year
    thermal_driving: float    # degC
    q_overturn: float         # m3/s
    q_total: float            # m3/s
    q_export: float           # m3/s, dense water leaving the shelf
    chi_dsw: float            # fraction of cavity inflow that is DSW, 0..1
    rho_shelf: float          # kg/m3
    rho_cdw: float            # kg/m3
    rho_cavity: float         # kg/m3
    T_inflow: float           # degC
    S_inflow: float           # g/kg


class CavityBoxModel:
    """Four-variable box model of one ice-shelf cavity.

    Examples
    --------
    >>> model = CavityBoxModel(BoxParams(CAVITIES["Filchner-Ronne"]))
    >>> x = model.initial_state("cold")
    >>> xs = model.steady_state(x)                        # doctest: +SKIP
    >>> model.diagnostics(xs).melt_rate                   # doctest: +SKIP
    """

    def __init__(self, params: BoxParams):
        self.p = params
        self.g = params.geom

    # -- internal pieces ---------------------------------------------------- #

    def _melt(self, T_c: float, S_c: float) -> tuple[float, float]:
        """Return ``(melt rate in m/s of ice, thermal driving in degC)``."""
        T_f = freezing_point(S_c, self.g.draft)
        td = T_c - T_f
        L_eff = latent_heat_effective(T_f, self.p.T_ice)
        m = self.p.gamma_T * td * (CONST.rho_sw * CONST.c_pw) / (CONST.rho_i * L_eff)
        return m, td

    def _inflow(self, T_s: float, S_s: float) -> tuple[float, float, float, float]:
        """Partition the cavity inflow between DSW and mWDW.

        The fraction of dense shelf water is a smooth function of the density
        contrast between the coastal box and the CDW reservoir, both evaluated
        at the ice-draft depth.  When the coastal box is denser, DSW sinks into
        the cavity; when it is lighter, mWDW ventilates the cavity instead.

        Returns ``(chi, T_in, S_in, drho_shelf)`` where ``drho_shelf`` is the
        coastal-to-CDW density excess that also drives the off-shelf export.
        """
        rho_s = density(T_s, S_s)
        rho_d = density(self.p.T_cdw, self.p.S_cdw)
        drho_shelf = rho_s - rho_d
        chi = smooth_step(drho_shelf, self.p.drho_switch)
        T_in = chi * T_s + (1.0 - chi) * self.p.T_cdw
        S_in = chi * S_s + (1.0 - chi) * self.p.S_cdw
        return chi, T_in, S_in, drho_shelf

    def _exchange(self, T_c: float, S_c: float, T_in: float, S_in: float,
                  drho_shelf: float) -> tuple[float, float]:
        """Return ``(overturning flux, total cavity ventilation flux)`` in m3/s.

        Two buoyancy sources drive the cavity overturning.  The *meltwater pump*
        scales with the density excess of the inflow over the cavity water and
        dominates in the warm regime.  The *dense-water gravity current* scales
        with the density excess of the coastal box over the CDW and dominates in
        the cold regime, where meltwater production is too weak to sustain a
        circulation on its own.  A background exchange set by the front mixing
        velocity operates in both regimes.
        """
        drho_pump = density(T_in, S_in) - density(T_c, S_c)
        q_ovt = (self.p.overturning_coefficient * softplus(drho_pump, self.p.drho_soft)
                 + self.p.gravity_current_coefficient * softplus(drho_shelf, self.p.drho_soft))
        q_mix = self.p.gamma_star * self.g.front_area
        return q_ovt, q_ovt + q_mix

    def _export(self, drho_shelf: float) -> float:
        """Off-shelf export of dense shelf water down the continental slope, m3/s.

        Without this pathway the salt delivered by brine rejection would simply
        accumulate in the coastal box and drive its salinity to unphysical
        values; in reality it is carried into the deep ocean as bottom water.
        """
        return self.p.export_coefficient * softplus(drho_shelf, self.p.drho_soft)

    # -- right-hand side ---------------------------------------------------- #

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """Time derivative of the state vector (units per second).

        ``t`` is accepted for compatibility with :func:`scipy.integrate.solve_ivp`
        and is unused unless a time-dependent forcing wrapper is applied.
        """
        T_c, S_c, T_s, S_s = x
        p, g = self.p, self.g

        m, _ = self._melt(T_c, S_c)
        chi, T_in, S_in, drho_shelf = self._inflow(T_s, S_s)
        _, q = self._exchange(T_c, S_c, T_in, S_in, drho_shelf)
        q_exp = self._export(drho_shelf)

        V_c, V_s = g.cavity_volume, g.shelf_volume
        A_c = g.area_ice

        # Freshwater volume flux released by melting, m3/s.
        F_melt = A_c * m * CONST.rho_i / CONST.rho_fw

        # --- cavity ---------------------------------------------------------
        # Heat removed by melting equals rho_sw c_pw gamma_T (T_c - T_f) per unit area.
        T_f = freezing_point(S_c, g.draft)
        dT_c = (q * (T_in - T_c) - A_c * p.gamma_T * (T_c - T_f)) / V_c
        dS_c = (q * (S_in - S_c) - F_melt * S_c) / V_c

        # --- coastal box ----------------------------------------------------
        # Brine rejection: sea-ice formation removes fresh water and leaves salt.
        brine = g.polynya_area * p.sigma_si * (CONST.rho_i / CONST.rho_fw) * (S_s - p.S_ice)

        # Dense water exported off the shelf is replaced by CDW, so the export
        # enters the budgets as an additional relaxation towards CDW properties.
        # Only a fraction of the buoyant cavity outflow is entrained into the
        # coastal box; the remainder leaves along the ice-shelf front.
        q_in = p.eps_out * q
        dT_s = (p.q_shelf * (p.T_cdw - T_s)
                + q_exp * (p.T_cdw - T_s)
                + q_in * (T_c - T_s)
                - g.polynya_area * p.lambda_atm * (T_s - p.T_surface)) / V_s
        dS_s = (p.q_shelf * (p.S_cdw - S_s)
                + q_exp * (p.S_cdw - S_s)
                + q_in * (S_c - S_s)
                + brine) / V_s

        return np.array([dT_c, dS_c, dT_s, dS_s])

    def jacobian(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Central-difference Jacobian ``dF/dx`` of :meth:`rhs`."""
        x = np.asarray(x, dtype=float)
        J = np.empty((NDIM, NDIM))
        for j in range(NDIM):
            h = eps * max(abs(x[j]), 1.0)
            xp, xm = x.copy(), x.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (self.rhs(0.0, xp) - self.rhs(0.0, xm)) / (2.0 * h)
        return J

    # -- states ------------------------------------------------------------- #

    def initial_state(self, regime: str = "cold") -> np.ndarray:
        """A plausible starting point on the cold or warm branch."""
        if regime == "cold":
            return np.array([-1.85, 34.60, -1.90, 34.75])
        if regime == "warm":
            return np.array([self.p.T_cdw - 0.2, 34.55, -1.0, 34.45])
        raise ValueError(f"regime must be 'cold' or 'warm', got {regime!r}")

    def steady_state(self, x0: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Solve ``rhs = 0`` from the initial guess ``x0``.

        Raises
        ------
        RuntimeError
            If the nonlinear solve does not converge.
        """
        from scipy.optimize import root

        # Scale the residuals so temperature and salinity balances are comparable.
        scale = np.array([1.0 / max(self.g.cavity_volume, 1.0)] * 2
                         + [1.0 / max(self.g.shelf_volume, 1.0)] * 2)
        scale = scale / scale.max()

        sol = root(lambda y: self.rhs(0.0, y) / scale, x0, method="hybr", tol=tol)
        if not sol.success:
            raise RuntimeError(f"steady state did not converge: {sol.message}")
        return sol.x

    def is_stable(self, x: np.ndarray) -> tuple[bool, np.ndarray]:
        """Linear stability of a fixed point: ``(stable, eigenvalues)``."""
        eig = np.linalg.eigvals(self.jacobian(x))
        return bool(np.all(eig.real < 0.0)), eig

    def equilibrium(self, regime: str = "cold", spinup_years: float = 8000.0,
                    residual_tol: float = 1e-9, require_stable: bool = True
                    ) -> np.ndarray:
        """Find the *attractor* of the requested regime.

        A plain Newton solve is not enough here.  Newton happily converges onto
        the unstable middle branch, returning a state that satisfies ``rhs = 0``
        but that the model would never occupy -- calibrating against such a
        point fits the wrong thing entirely.  This routine therefore integrates
        forward from the regime's initial guess to identify the attractor the
        system actually settles on, polishes it with Newton, and only accepts it
        if it is linearly stable and still in the requested regime.

        Parameters
        ----------
        require_stable
            When ``True`` (the default) a converged but unstable fixed point is
            rejected, as is an attractor belonging to the other regime.

        Raises
        ------
        RuntimeError
            If the requested regime is not an attractor for these parameters.
        """
        x0 = self.initial_state(regime)
        scale = np.array([1.0 / self.g.cavity_volume] * 2
                         + [1.0 / self.g.shelf_volume] * 2)
        scale = scale / scale.max()

        _, X = self.integrate(x0, years=spinup_years, n_out=8)
        x = X[-1]
        try:
            x_polished = self.steady_state(x)
            if np.all(np.isfinite(x_polished)) and \
                    np.max(np.abs(self.rhs(0.0, x_polished) / scale)) < residual_tol:
                x = x_polished
        except RuntimeError:
            pass

        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"{self.g.name}: integration diverged in the {regime} regime")

        if require_stable:
            stable, _ = self.is_stable(x)
            if not stable:
                raise RuntimeError(f"{self.g.name}: no stable {regime} state")
            chi = self.diagnostics(x).chi_dsw
            if (regime == "cold" and chi < 0.5) or (regime == "warm" and chi > 0.5):
                raise RuntimeError(
                    f"{self.g.name}: {regime} initial condition relaxes to the "
                    f"other regime (chi = {chi:.2f})")

        return x

    # -- diagnostics -------------------------------------------------------- #

    def diagnostics(self, x: np.ndarray) -> Diagnostics:
        """Physical diagnostics at state ``x``."""
        T_c, S_c, T_s, S_s = x
        m, td = self._melt(T_c, S_c)
        chi, T_in, S_in, drho_shelf = self._inflow(T_s, S_s)
        q_ovt, q_tot = self._exchange(T_c, S_c, T_in, S_in, drho_shelf)

        melt_m_per_yr = m * SEC_PER_YEAR
        # Gt/yr: m/s * m2 * kg/m3 * s/yr / 1e12 kg per Gt
        melt_gt_per_yr = m * self.g.area_ice * CONST.rho_i * SEC_PER_YEAR / 1e12

        return Diagnostics(
            melt_rate=melt_m_per_yr,
            melt_flux=melt_gt_per_yr,
            thermal_driving=td,
            q_overturn=q_ovt,
            q_total=q_tot,
            q_export=self._export(drho_shelf),
            chi_dsw=chi,
            rho_shelf=density(T_s, S_s),
            rho_cdw=density(self.p.T_cdw, self.p.S_cdw),
            rho_cavity=density(T_c, S_c),
            T_inflow=T_in,
            S_inflow=S_in,
        )

    # -- time integration --------------------------------------------------- #

    def integrate(self, x0: np.ndarray, years: float, n_out: int = 1000,
                  forcing: Callable[[float], dict] | None = None,
                  noise: float = 0.0, seed: int | None = None,
                  rtol: float = 1e-8, atol: float = 1e-10):
        """Integrate the model forward in time.

        Parameters
        ----------
        x0
            Initial state.
        years
            Integration length in years.
        n_out
            Number of evenly spaced output times.
        forcing
            Optional callable mapping time in years to a dict of control
            parameters to override, e.g. ``lambda t: {"T_cdw": 0.5 + 0.01 * t}``.
            Used for the rate-induced tipping experiments.
        noise
            Standard deviation of additive white noise on temperature
            (degC per sqrt(year)); when non-zero an Euler-Maruyama scheme is
            used instead of the adaptive solver, as required for the
            critical-slowing-down diagnostics.
        seed
            Seed for the noise realisation.

        Returns
        -------
        t : ndarray
            Output times in years.
        X : ndarray, shape (n_out, 4)
            State trajectory.
        """
        if noise > 0.0:
            return self._integrate_stochastic(x0, years, n_out, forcing, noise, seed)

        from scipy.integrate import solve_ivp

        base = self.p
        t_end = years * SEC_PER_YEAR
        t_eval = np.linspace(0.0, t_end, n_out)

        def f(t, y):
            if forcing is not None:
                self.p = base.with_control(**forcing(t / SEC_PER_YEAR))
            try:
                return self.rhs(t, y)
            finally:
                self.p = base

        sol = solve_ivp(f, (0.0, t_end), np.asarray(x0, float), t_eval=t_eval,
                        method="LSODA", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")
        return sol.t / SEC_PER_YEAR, sol.y.T

    def _integrate_stochastic(self, x0, years, n_out, forcing, noise, seed):
        """Euler-Maruyama integration with additive noise on the temperatures."""
        rng = np.random.default_rng(seed)
        base = self.p

        # Resolve the fastest box turnover time.
        dt_year = min(0.02, years / (20.0 * n_out))
        n_steps = int(np.ceil(years / dt_year))
        dt = dt_year * SEC_PER_YEAR
        stride = max(1, n_steps // n_out)

        # Noise acts on temperature only (surface heat-flux variability).
        amp = np.array([noise, 0.0, noise, 0.0]) / np.sqrt(SEC_PER_YEAR)

        x = np.asarray(x0, dtype=float).copy()
        ts, xs = [0.0], [x.copy()]
        for k in range(1, n_steps + 1):
            t_year = k * dt_year
            if forcing is not None:
                self.p = base.with_control(**forcing(t_year))
            try:
                x = x + self.rhs(k * dt, x) * dt + amp * np.sqrt(dt) * rng.standard_normal(NDIM)
            finally:
                self.p = base
            if k % stride == 0:
                ts.append(t_year)
                xs.append(x.copy())

        return np.asarray(ts), np.asarray(xs)


# --------------------------------------------------------------------------- #
# Convenience constructors
# --------------------------------------------------------------------------- #

def make_model(cavity: str, **controls: float) -> CavityBoxModel:
    """Build a model for a named cavity with optional control-parameter overrides."""
    if cavity not in CAVITIES:
        raise KeyError(f"unknown cavity {cavity!r}; known: {sorted(CAVITIES)}")
    return CavityBoxModel(BoxParams(CAVITIES[cavity], **controls))
