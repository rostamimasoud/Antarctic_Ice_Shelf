"""Tests for the low-dimensional cavity model and its continuation."""

from __future__ import annotations

import numpy as np
import pytest

from aisgnn.boxmodel.cavity import (
    CAVITIES,
    BoxParams,
    CavityBoxModel,
    density,
    freezing_point,
    make_model,
    smooth_step,
    softplus,
)
from aisgnn.boxmodel.continuation import bifurcation_diagram, continue_branch, find_folds
from aisgnn.config import CONST


# --------------------------------------------------------------------------- #
# Thermodynamics
# --------------------------------------------------------------------------- #

def test_freezing_point_falls_with_depth():
    """Pressure lowers the freezing point; getting this sign wrong inverts melt."""
    assert freezing_point(34.7, -500.0) < freezing_point(34.7, 0.0)


def test_freezing_point_surface_value():
    """Surface freezing point of typical seawater is about -1.9 degC."""
    assert freezing_point(34.5, 0.0) == pytest.approx(-1.89, abs=0.02)


def test_freezing_point_falls_with_salinity():
    assert freezing_point(35.0, -300.0) < freezing_point(34.0, -300.0)


def test_density_monotonicity():
    assert density(-1.0, 34.8) > density(-1.0, 34.5)     # saltier is denser
    assert density(-1.9, 34.7) > density(1.0, 34.7)      # colder is denser


def test_smooth_step_limits():
    assert smooth_step(-10.0, 0.05) == pytest.approx(0.0, abs=1e-6)
    assert smooth_step(10.0, 0.05) == pytest.approx(1.0, abs=1e-6)
    assert smooth_step(0.0, 0.05) == pytest.approx(0.5)


def test_softplus_matches_relu_away_from_zero():
    assert softplus(1.0, 1e-3) == pytest.approx(1.0, rel=1e-6)
    assert softplus(-1.0, 1e-3) == pytest.approx(0.0, abs=1e-6)
    assert softplus(0.0, 1e-3) > 0.0                      # smooth, never exactly zero


# --------------------------------------------------------------------------- #
# Model structure
# --------------------------------------------------------------------------- #

def test_state_dimension():
    m = make_model("Filchner-Ronne")
    assert m.rhs(0.0, m.initial_state("cold")).shape == (4,)


def test_unknown_cavity_raises():
    with pytest.raises(KeyError):
        make_model("Atlantis")


def test_melt_sign_follows_thermal_driving():
    """Water above the local freezing point melts; below it, ice accretes."""
    m = make_model("Pine Island", T_cdw=1.2)
    T_f = freezing_point(34.5, CAVITIES["Pine Island"].draft)
    melt_warm, td_warm = m._melt(T_f + 1.0, 34.5)
    melt_cold, td_cold = m._melt(T_f - 0.1, 34.5)
    assert td_warm > 0 and melt_warm > 0
    assert td_cold < 0 and melt_cold < 0


def test_melt_magnitude_is_physical():
    """A degree of thermal driving gives melt of order a metre per year."""
    m = make_model("Pine Island")
    T_f = freezing_point(34.5, CAVITIES["Pine Island"].draft)
    rate, _ = m._melt(T_f + 1.0, 34.5)
    assert 0.5 < rate * CONST.sec_per_year < 20.0


def test_equilibrium_is_a_fixed_point_and_stable():
    m = make_model("Filchner-Ronne", sigma=8.0, T_cdw=0.5)
    x = m.equilibrium("cold")
    scale = np.array([1.0 / m.g.cavity_volume] * 2 + [1.0 / m.g.shelf_volume] * 2)
    scale = scale / scale.max()
    assert np.max(np.abs(m.rhs(0.0, x) / scale)) < 1e-9
    stable, eig = m.is_stable(x)
    assert stable and np.all(eig.real < 0.0)


def test_equilibrium_returns_the_requested_regime():
    """Whatever is returned must actually belong to the regime that was asked for.

    Cavities for which the regime is not an attractor raise instead of quietly
    handing back the other branch, which is what makes calibration meaningful.
    """
    checked = 0
    for cavity in ("Filchner-Ronne", "Ross", "Pine Island", "Getz", "Thwaites"):
        for sigma, regime in ((10.0, "cold"), (1.0, "warm")):
            m = make_model(cavity, sigma=sigma, T_cdw=0.8)
            try:
                x = m.equilibrium(regime)
            except RuntimeError:
                continue                      # regime genuinely absent: acceptable
            chi = m.diagnostics(x).chi_dsw
            assert (chi > 0.5) == (regime == "cold"), \
                f"{cavity} {regime} returned chi={chi:.2f}"
            checked += 1
    assert checked > 0, "no equilibria found at all"


def test_equilibrium_rejects_unstable_fixed_points():
    """A fixed point on the unstable middle branch must never be returned.

    Newton converges onto it happily; accepting it would mean calibrating the
    model against a state it can never occupy.
    """
    for cavity in ("Filchner-Ronne", "Ross", "Getz", "Pine Island"):
        for sigma in (0.5, 2.0, 6.0, 12.0):
            for regime in ("cold", "warm"):
                m = make_model(cavity, sigma=sigma, T_cdw=0.8)
                try:
                    x = m.equilibrium(regime)
                except RuntimeError:
                    continue
                stable, _ = m.is_stable(x)
                assert stable, f"{cavity} sigma={sigma} {regime} returned an unstable state"


def test_diagnostics_are_self_consistent():
    """Unit conversions and flux orderings hold whichever regime the cavity is in."""
    m = make_model("Pine Island", T_cdw=1.2, sigma=4.0)
    x = next(m.equilibrium(r) for r in ("warm", "cold")
             if _has_regime(m, r))
    d = m.diagnostics(x)
    area = CAVITIES["Pine Island"].area_ice
    expected_gt = d.melt_rate * area * CONST.rho_i / 1e12
    assert d.melt_flux == pytest.approx(expected_gt, rel=1e-9)
    assert 0.0 <= d.chi_dsw <= 1.0
    assert d.q_total >= d.q_overturn


def _has_regime(model, regime: str) -> bool:
    try:
        model.equilibrium(regime)
    except RuntimeError:
        return False
    return True


def test_jacobian_matches_finite_difference_of_rhs():
    m = make_model("Ross", sigma=10.0)
    x = m.equilibrium("cold")
    J = m.jacobian(x)
    v = np.array([1e-4, 1e-5, 1e-4, 1e-5])
    got = (m.rhs(0.0, x + v) - m.rhs(0.0, x - v)) / 2.0
    assert np.allclose(J @ v, got, rtol=1e-3, atol=1e-14)


# --------------------------------------------------------------------------- #
# Bistability
# --------------------------------------------------------------------------- #

def test_bistability_exists_somewhere_in_sigma():
    """Two distinct attractors must coexist for some sea-ice formation rate.

    Without this the model has no tipping point to study at all.
    """
    found = False
    for sigma in (1.0, 2.0, 3.0, 4.0, 5.0):
        m = make_model("Filchner-Ronne", sigma=sigma, T_cdw=0.5)
        try:
            cold = m.diagnostics(m.equilibrium("cold")).melt_rate
            warm = m.diagnostics(m.equilibrium("warm")).melt_rate
        except RuntimeError:
            continue
        if abs(warm - cold) > 0.5:
            found = True
            break
    assert found, "no bistable window found in sigma"


def test_warm_state_melts_more_than_cold_state():
    m = make_model("Filchner-Ronne", sigma=3.0, T_cdw=0.5)
    cold = m.diagnostics(m.equilibrium("cold"))
    warm = m.diagnostics(m.equilibrium("warm"))
    assert warm.melt_rate > cold.melt_rate
    assert warm.chi_dsw < cold.chi_dsw      # warm cavity is not DSW-ventilated


# --------------------------------------------------------------------------- #
# Continuation
# --------------------------------------------------------------------------- #

def test_continuation_resolves_an_unstable_branch():
    """The S-curve must include an unstable middle branch.

    If the corrector jumps between branches the curve looks continuous but has
    no unstable segment, which is the failure this guards against.
    """
    base = BoxParams(CAVITIES["Filchner-Ronne"], T_cdw=0.5)
    branch, folds, hyst = bifurcation_diagram(
        base, "sigma", p_start=2.0, p_min=0.0, p_max=15.0, regime="cold")

    assert len(branch) > 50
    assert (~branch.stable).sum() > 5, "no unstable branch resolved"
    assert len(folds) >= 2
    assert hyst is not None and hyst.width > 0.0
    assert hyst.p_reverse < hyst.p_forward


def test_folds_lie_at_turning_points():
    """Each detected fold must sit at a reversal in the control parameter."""
    base = BoxParams(CAVITIES["Filchner-Ronne"], T_cdw=0.5)
    branch, folds, _ = bifurcation_diagram(
        base, "sigma", p_start=2.0, p_min=0.0, p_max=15.0, regime="cold")

    for fold in folds:
        assert branch.p.min() - 0.1 <= fold.p <= branch.p.max() + 0.1
        assert fold.residual < 1e-6
        assert fold.direction in {"cold_to_warm", "warm_to_cold"}


def test_continuation_stays_on_branch_when_monostable():
    """Far above the bistable window the branch is monostable and fold-free."""
    base = BoxParams(CAVITIES["Filchner-Ronne"], T_cdw=0.5)
    m = CavityBoxModel(base.with_control(sigma=12.0))
    x0 = m.equilibrium("cold")
    branch = continue_branch(base, "sigma", x0, 12.0, 10.0, 20.0, direction=+1)
    assert branch.stable.all()
    assert find_folds(base, branch) == []
