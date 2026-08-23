"""Tests for the reduced flowline ice model."""

from __future__ import annotations

import numpy as np
import pytest

from aisgnn.coupling.flowline import (
    FlowlineConfig,
    calibrate_flux,
    compare_histories,
    constant_melt,
    flotation_thickness,
    gradual_melt,
    grounding_line_flux,
    integrate_flowline,
    regime_shift_melt,
)

PROGRADE = FlowlineConfig(bed_slope=-1.0e-3, bed_depth_at_origin=-400.0)
RETROGRADE = FlowlineConfig(bed_slope=+1.0e-3, bed_depth_at_origin=-900.0)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def test_retrograde_flag_matches_deepening_inland():
    """Retreat is towards smaller x, so a positive slope deepens inland."""
    assert RETROGRADE.retrograde
    assert not PROGRADE.retrograde
    # Confirm directly from the bed rather than from the flag alone.
    assert RETROGRADE.bed(200e3) < RETROGRADE.bed(300e3)
    assert PROGRADE.bed(200e3) > PROGRADE.bed(300e3)


def test_flotation_thickness_scales_with_depth():
    """Thickness at flotation is the bed depth times the density ratio.

    Compared within one configuration: which of the two beds is deeper at a
    given x depends on both slope and offset, so comparing across them tests an
    accident of the chosen constants rather than the physics.
    """
    deep = FlowlineConfig(bed_slope=0.0, bed_depth_at_origin=-900.0)
    shallow = FlowlineConfig(bed_slope=0.0, bed_depth_at_origin=-400.0)
    assert flotation_thickness(300e3, deep) > flotation_thickness(300e3, shallow)

    expected = 400.0 * 1028.0 / 917.0
    assert flotation_thickness(300e3, shallow) == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def test_calibration_puts_the_initial_state_in_balance():
    """The control run must not drift; otherwise every result is contaminated."""
    for cfg in (PROGRADE, RETROGRADE):
        k = calibrate_flux(cfg)
        h0 = flotation_thickness(cfg.x_init, cfg)
        q = grounding_line_flux(h0, cfg.melt_reference, k, cfg)
        assert q == pytest.approx(cfg.accumulation * cfg.x_init, rel=1e-9)


def test_control_run_does_not_drift():
    res = integrate_flowline(PROGRADE, constant_melt(PROGRADE.melt_reference),
                             years=400.0)
    assert abs(res.total_retreat) < 1.0          # metres over four centuries
    assert res.sea_level[-1] == pytest.approx(0.0, abs=1e-6)
    assert not res.collapsed


# --------------------------------------------------------------------------- #
# Flux law
# --------------------------------------------------------------------------- #

def test_flux_increases_with_thickness_and_melt():
    k = calibrate_flux(PROGRADE)
    base = grounding_line_flux(800.0, 1.0, k, PROGRADE)
    assert grounding_line_flux(900.0, 1.0, k, PROGRADE) > base
    assert grounding_line_flux(800.0, 5.0, k, PROGRADE) > base


def test_flux_never_negative_under_extreme_reduction():
    """A large melt decrease must not drive the buttressing factor below zero."""
    k = calibrate_flux(PROGRADE)
    assert grounding_line_flux(800.0, -100.0, k, PROGRADE) >= 0.0


# --------------------------------------------------------------------------- #
# Response to melt
# --------------------------------------------------------------------------- #

def test_more_melt_causes_retreat():
    k = calibrate_flux(PROGRADE)
    res = integrate_flowline(PROGRADE, constant_melt(8.0), years=300.0, k=k)
    assert res.total_retreat > 0.0
    assert res.sea_level[-1] > 0.0


def test_retrograde_bed_collapses_where_prograde_does_not():
    """The marine ice sheet instability must appear only on the retrograde bed."""
    pro = compare_histories(PROGRADE, before=1.0, after=12.0, years=400.0)
    retro = compare_histories(RETROGRADE, before=1.0, after=12.0, years=400.0)
    assert not pro["gradual"].collapsed
    assert retro["gradual"].collapsed


def test_abrupt_melt_causes_more_retreat_than_gradual():
    """Same endpoint melt: any difference is caused by the pace of the change."""
    out = compare_histories(PROGRADE, before=1.0, after=12.0, t_shift=100.0,
                            years=400.0)
    assert out["abrupt"].total_retreat > out["gradual"].total_retreat
    assert out["abrupt"].sea_level[-1] > out["gradual"].sea_level[-1]
    assert out["control"].total_retreat == pytest.approx(0.0, abs=1.0)


def test_sea_level_is_monotonic():
    """Cumulative contribution can only increase."""
    res = integrate_flowline(PROGRADE, constant_melt(8.0), years=300.0)
    assert np.all(np.diff(res.sea_level) >= -1e-12)


def test_grounding_line_stops_at_the_domain_limit():
    res = integrate_flowline(RETROGRADE, constant_melt(30.0), years=600.0)
    assert res.collapsed
    assert res.position.min() >= RETROGRADE.x_min - 1e-9
    assert res.time_of_collapse is not None


# --------------------------------------------------------------------------- #
# Melt histories
# --------------------------------------------------------------------------- #

def test_melt_histories_reach_their_endpoints():
    ramp = gradual_melt(1.0, 10.0, 200.0)
    assert ramp(0.0) == pytest.approx(1.0)
    assert ramp(200.0) == pytest.approx(10.0)
    assert ramp(400.0) == pytest.approx(10.0)      # held after the ramp

    shift = regime_shift_melt(1.0, 10.0, 100.0, width=2.0)
    assert shift(0.0) == pytest.approx(1.0, abs=0.05)
    assert shift(100.0) == pytest.approx(5.5, abs=0.05)
    assert shift(200.0) == pytest.approx(10.0, abs=0.05)


def test_regime_shift_is_sharper_than_the_ramp():
    shift = regime_shift_melt(1.0, 10.0, 100.0, width=2.0)
    ramp = gradual_melt(1.0, 10.0, 400.0)
    t = np.linspace(0.0, 400.0, 401)
    assert np.max(np.abs(np.gradient([shift(x) for x in t], t))) > \
           np.max(np.abs(np.gradient([ramp(x) for x in t], t)))
