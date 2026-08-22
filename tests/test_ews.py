"""Tests for the early-warning-signal estimators.

These check the estimators against processes whose answers are known
analytically, and -- as importantly -- that they do *not* fire on null data.
An indicator that reports critical slowing down on stationary noise would
invalidate every result built on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from aisgnn.dynsys.ews import (
    compute_ews,
    detrend,
    indicator_sensitivity,
    phase_randomised_surrogate,
    rolling_autocorrelation,
    rolling_recovery_rate,
    rolling_variance,
    time_irreversibility,
    trend_significance,
)


def ar1(n: int, rho: float, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """An AR(1) process with known lag-1 autocorrelation ``rho``."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma / np.sqrt(1.0 - rho ** 2))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0.0, sigma)
    return x


# --------------------------------------------------------------------------- #
# Estimator correctness
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8, 0.95])
def test_autocorrelation_recovers_known_ar1_coefficient(rho):
    x = ar1(40000, rho)
    _, ac = rolling_autocorrelation(x, window=20000, step=20000)
    assert ac[0] == pytest.approx(rho, abs=0.03)


def test_variance_recovers_known_ar1_variance():
    rho, sigma = 0.5, 1.0
    x = ar1(80000, rho, sigma)
    _, var = rolling_variance(x, window=80000)
    assert var[0] == pytest.approx(sigma ** 2 / (1 - rho ** 2), rel=0.05)


def test_recovery_rate_is_negative_and_rises_towards_zero():
    """Slower recovery means a leading rate closer to zero from below."""
    _, slow = rolling_recovery_rate(ar1(40000, 0.95), window=40000)
    _, fast = rolling_recovery_rate(ar1(40000, 0.50), window=40000)
    assert slow[0] < 0.0 and fast[0] < 0.0
    assert slow[0] > fast[0]


def test_irreversibility_vanishes_for_a_reversible_process():
    """Gaussian AR(1) is time-reversible, so the statistic must be near zero."""
    assert abs(time_irreversibility(ar1(100000, 0.7))) < 0.05


def test_irreversibility_detects_an_asymmetric_process():
    """Slow build-up with fast collapse is strongly irreversible."""
    rng = np.random.default_rng(1)
    x = np.zeros(20000)
    for i in range(1, x.size):
        x[i] = x[i - 1] + 0.01 + rng.normal(0, 0.02)
        if x[i] > 1.0:
            x[i] = 0.0                       # sawtooth: gradual rise, sharp drop
    assert abs(time_irreversibility(x)) > 0.2


# --------------------------------------------------------------------------- #
# Detrending
# --------------------------------------------------------------------------- #

def test_detrend_removes_a_slow_ramp_but_keeps_fluctuations():
    n = 4000
    trend = np.linspace(0.0, 10.0, n)
    noise = ar1(n, 0.5, 1.0, seed=3)
    resid = detrend(trend + noise, bandwidth=100)
    assert abs(np.polyfit(np.arange(n), resid, 1)[0]) < 1e-3
    assert resid.std() == pytest.approx(noise.std(), rel=0.25)


def test_detrend_does_not_manufacture_edge_variance():
    """Edge padding must not suppress the ends and fake a variance trend."""
    x = ar1(3000, 0.6, seed=4)
    resid = detrend(x, bandwidth=100)
    edge = np.r_[resid[:200], resid[-200:]].var()
    middle = resid[1200:1800].var()
    assert 0.4 < edge / middle < 2.5


def test_surrogate_preserves_the_power_spectrum():
    x = ar1(4096, 0.8, seed=5)
    rng = np.random.default_rng(0)
    s = phase_randomised_surrogate(x, rng)
    px = np.abs(np.fft.rfft(x - x.mean()))
    ps = np.abs(np.fft.rfft(s - s.mean()))
    assert np.allclose(px, ps, rtol=1e-6, atol=1e-6)
    assert s.size == x.size


# --------------------------------------------------------------------------- #
# Trend detection: it must fire on signal and stay silent on noise
# --------------------------------------------------------------------------- #

def test_no_false_alarm_on_stationary_noise():
    """The crucial null: stationary AR(1) must not yield a significant trend."""
    res = compute_ews(ar1(3000, 0.5, seed=7), window=300, n_surrogates=200)
    assert not res.trends["variance"].significant
    assert not res.trends["autocorrelation"].significant


def test_detects_slowing_down_in_a_ramped_ar1_process():
    """A process whose autocorrelation is driven towards one must be detected."""
    rng = np.random.default_rng(11)
    n = 3000
    rho = np.linspace(0.30, 0.985, n)
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = rho[i] * x[i - 1] + rng.normal(0.0, 1.0)

    res = compute_ews(x, window=300, n_surrogates=200)
    assert res.trends["autocorrelation"].tau > 0.4
    assert res.trends["autocorrelation"].significant
    assert res.trends["variance"].significant


def test_surrogate_p_value_is_stricter_than_the_naive_one():
    """Rolling windows autocorrelate the indicator, so the naive test overstates."""
    res = compute_ews(ar1(3000, 0.7, seed=13), window=300, n_surrogates=200)
    t = res.trends["variance"]
    if np.isfinite(t.p_value) and np.isfinite(t.p_value_naive):
        assert t.p_value >= t.p_value_naive - 1e-9


def test_trend_test_handles_degenerate_input():
    out = trend_significance(np.array([np.nan, np.nan]), n_surrogates=10)
    assert not out.significant and out.n_surrogates == 0


# --------------------------------------------------------------------------- #
# Robustness reporting
# --------------------------------------------------------------------------- #

def test_indicator_sensitivity_spans_the_requested_grid():
    x = ar1(2000, 0.6, seed=17)
    grid = indicator_sensitivity(x, windows=(200, 400), bandwidths=(50, 100))
    assert set(grid) == {(200, 50), (200, 100), (400, 50), (400, 100)}
    assert all(np.isfinite(v) for v in grid.values())
