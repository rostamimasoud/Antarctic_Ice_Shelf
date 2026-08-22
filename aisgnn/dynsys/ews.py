"""Early-warning signals of an approaching bifurcation.

As a system loses resilience near a saddle-node its leading eigenvalue
approaches zero, perturbations decay more slowly, and the residual fluctuations
become larger and more autocorrelated.  This module implements the standard
critical-slowing-down indicators together with a time-irreversibility statistic,
and -- importantly -- the significance test that decides whether an apparent
trend means anything.

Two practical points govern the implementation:

* **Detrending is not optional and not innocent.** Variance and autocorrelation
  are only interpretable once the slow forced drift is removed, but too narrow a
  filter removes the signal itself.  The bandwidth is therefore an explicit
  argument, and :func:`indicator_sensitivity` reports how much the answer
  depends on it.
* **Trends need a null.** Rolling-window indicators are strongly autocorrelated
  by construction, so the ordinary Kendall-tau p-value is far too permissive.
  Significance is assessed against phase-randomised surrogates that preserve the
  power spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

# --------------------------------------------------------------------------- #
# Detrending
# --------------------------------------------------------------------------- #

def detrend(x: np.ndarray, bandwidth: int, method: str = "gaussian") -> np.ndarray:
    """Remove the slow component of ``x``, returning the residual.

    Parameters
    ----------
    x
        Time series.
    bandwidth
        Filter width in samples.  For ``gaussian`` this is the standard
        deviation; for ``linear`` it is ignored.
    method
        ``gaussian`` (default), ``linear``, or ``none``.
    """
    x = np.asarray(x, float)
    if method == "none":
        return x - x.mean()
    if method == "linear":
        return signal.detrend(x, type="linear")
    if method != "gaussian":
        raise ValueError(f"unknown detrend method {method!r}")

    if bandwidth < 1:
        raise ValueError("bandwidth must be at least one sample")

    # Reflect at the edges so the filter does not pull the ends towards zero,
    # which would manufacture a variance trend exactly where it is being looked
    # for.
    pad = min(3 * bandwidth, x.size - 1)
    padded = np.pad(x, pad, mode="reflect")
    kernel = signal.windows.gaussian(6 * bandwidth + 1, bandwidth)
    kernel /= kernel.sum()
    smooth = np.convolve(padded, kernel, mode="same")[pad:pad + x.size]
    return x - smooth


# --------------------------------------------------------------------------- #
# Rolling indicators
# --------------------------------------------------------------------------- #

def _strided(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """Stack sliding windows into a ``(n_windows, window)`` view.

    The indicators are recomputed on hundreds of surrogates per significance
    test, so the naive Python loop over windows dominates the runtime; this
    turns each indicator into a single vectorised reduction.  The result is a
    view, not a copy.
    """
    if window > x.size:
        raise ValueError(f"window {window} longer than series {x.size}")
    view = np.lib.stride_tricks.sliding_window_view(x, window)
    return view[::step]


def _window_index(n: int, window: int, step: int = 1) -> np.ndarray:
    """Index of the last sample in each window."""
    return np.arange(window - 1, n, step)[:((n - window) // step) + 1]


def rolling_variance(x: np.ndarray, window: int, step: int = 1
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Variance in a sliding window.  Returns ``(index, variance)``."""
    x = np.asarray(x, float)
    w = _strided(x, window, step)
    return _window_index(x.size, window, step), w.var(axis=1, ddof=1)


def rolling_autocorrelation(x: np.ndarray, window: int, lag: int = 1,
                            step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Lag-``lag`` autocorrelation in a sliding window."""
    x = np.asarray(x, float)
    w = _strided(x, window, step)
    centred = w - w.mean(axis=1, keepdims=True)
    denom = np.einsum("ij,ij->i", centred, centred)
    numer = np.einsum("ij,ij->i", centred[:, :-lag], centred[:, lag:])
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(denom > 0, numer / denom, np.nan)
    return _window_index(x.size, window, step), rho


def rolling_recovery_rate(x: np.ndarray, window: int, dt: float = 1.0,
                          step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Decay rate ``lambda`` from an AR(1) fit in a sliding window.

    For an Ornstein-Uhlenbeck process the lag-1 autocorrelation is
    ``exp(lambda dt)`` with ``lambda < 0``; ``lambda`` rising towards zero is the
    direct expression of critical slowing down.
    """
    idx, rho = rolling_autocorrelation(x, window, lag=1, step=step)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(rho > 0, np.log(rho) / dt, np.nan)
    return idx, lam


def rolling_skewness(x: np.ndarray, window: int, step: int = 1
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Skewness in a sliding window; it grows as the state nears a fold."""
    x = np.asarray(x, float)
    w = _strided(x, window, step)
    centred = w - w.mean(axis=1, keepdims=True)
    m2 = np.mean(centred ** 2, axis=1)
    m3 = np.mean(centred ** 3, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(m2 > 0, m3 / m2 ** 1.5, 0.0)
    return _window_index(x.size, window, step), skew


# --------------------------------------------------------------------------- #
# Time irreversibility
# --------------------------------------------------------------------------- #

def time_irreversibility(x: np.ndarray, lag: int = 1) -> float:
    """Normalised third-order asymmetry of the increments.

    For a time-reversible stationary process the increment distribution is
    symmetric and this statistic vanishes.  A saddle-node approach breaks that
    symmetry because excursions towards the fold decay differently from
    excursions away from it.
    """
    x = np.asarray(x, float)
    d = x[lag:] - x[:-lag]
    var = np.mean(d ** 2)
    if var <= 0:
        return 0.0
    return float(np.mean(d ** 3) / var ** 1.5)


def rolling_irreversibility(x: np.ndarray, window: int, lag: int = 1,
                            step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Time irreversibility in a sliding window."""
    x = np.asarray(x, float)
    w = _strided(x, window, step)
    d = w[:, lag:] - w[:, :-lag]
    var = np.mean(d ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(var > 0, np.mean(d ** 3, axis=1) / var ** 1.5, 0.0)
    return _window_index(x.size, window, step), out


# --------------------------------------------------------------------------- #
# Trend significance
# --------------------------------------------------------------------------- #

def phase_randomised_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A surrogate with the same power spectrum but randomised phases."""
    x = np.asarray(x, float)
    spectrum = np.fft.rfft(x - x.mean())
    phases = rng.uniform(0.0, 2.0 * np.pi, spectrum.size)
    phases[0] = 0.0
    if x.size % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=x.size) + x.mean()


@dataclass
class TrendTest:
    """Kendall-tau trend with a surrogate-based p-value."""

    tau: float
    p_value: float           # against phase-randomised surrogates
    p_value_naive: float     # the ordinary Kendall test, for comparison
    n_surrogates: int
    significant: bool


def trend_significance(indicator: np.ndarray, series: np.ndarray | None = None,
                       window: int | None = None, n_surrogates: int = 1000,
                       alpha: float = 0.05, seed: int | None = 0,
                       kind: str = "variance") -> TrendTest:
    """Test whether an indicator trends upwards more than chance allows.

    The null is built by recomputing the *same* indicator on phase-randomised
    surrogates of the original series, so the strong autocorrelation that
    rolling windows introduce is present under the null as well.  Passing only
    ``indicator`` falls back to shuffling it directly, which is weaker; supply
    ``series`` and ``window`` where possible.

    Parameters
    ----------
    indicator
        The rolling indicator whose trend is in question.
    series, window
        The residual series and window the indicator was computed from.
    kind
        Which indicator to recompute on the surrogates: ``variance``,
        ``autocorrelation``, ``recovery_rate``, ``skewness`` or
        ``irreversibility``.
    """
    indicator = np.asarray(indicator, float)
    good = np.isfinite(indicator)
    if good.sum() < 3:
        return TrendTest(np.nan, np.nan, np.nan, 0, False)

    t = np.arange(indicator.size)[good]
    tau, p_naive = stats.kendalltau(t, indicator[good])

    rng = np.random.default_rng(seed)
    fn = {"variance": rolling_variance,
          "autocorrelation": rolling_autocorrelation,
          "recovery_rate": rolling_recovery_rate,
          "skewness": rolling_skewness,
          "irreversibility": rolling_irreversibility}[kind]

    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        if series is not None and window is not None:
            _, surr_ind = fn(phase_randomised_surrogate(np.asarray(series, float), rng),
                             window)
        else:
            surr_ind = rng.permutation(indicator[good])
        ok = np.isfinite(surr_ind)
        if ok.sum() < 3:
            null[i] = 0.0
            continue
        null[i] = stats.kendalltau(np.arange(surr_ind.size)[ok], surr_ind[ok])[0]

    p_surr = float((null >= tau).mean())
    return TrendTest(tau=float(tau), p_value=p_surr, p_value_naive=float(p_naive),
                     n_surrogates=n_surrogates, significant=p_surr < alpha)


# --------------------------------------------------------------------------- #
# Convenience driver
# --------------------------------------------------------------------------- #

@dataclass
class EWSResult:
    """All indicators computed on one series."""

    index: np.ndarray
    variance: np.ndarray
    autocorrelation: np.ndarray
    recovery_rate: np.ndarray
    skewness: np.ndarray
    irreversibility: np.ndarray
    residual: np.ndarray
    trends: dict[str, TrendTest]


def compute_ews(x: np.ndarray, window: int, bandwidth: int | None = None,
                dt: float = 1.0, step: int = 1, detrend_method: str = "gaussian",
                n_surrogates: int = 200, seed: int | None = 0) -> EWSResult:
    """Compute every indicator on ``x`` and test each for a rising trend.

    ``bandwidth`` defaults to half the window, a common compromise between
    removing the forced drift and preserving the fluctuations being measured.
    """
    x = np.asarray(x, float)
    bandwidth = bandwidth if bandwidth is not None else max(window // 2, 1)
    resid = detrend(x, bandwidth, method=detrend_method)

    idx, var = rolling_variance(resid, window, step)
    _, ac = rolling_autocorrelation(resid, window, 1, step)
    _, lam = rolling_recovery_rate(resid, window, dt, step)
    _, skw = rolling_skewness(resid, window, step)
    _, irr = rolling_irreversibility(resid, window, 1, step)

    trends = {
        name: trend_significance(vals, series=resid, window=window,
                                 n_surrogates=n_surrogates, seed=seed, kind=name)
        for name, vals in (("variance", var), ("autocorrelation", ac),
                           ("recovery_rate", lam), ("skewness", skw),
                           ("irreversibility", irr))
    }

    return EWSResult(index=idx, variance=var, autocorrelation=ac,
                     recovery_rate=lam, skewness=skw, irreversibility=irr,
                     residual=resid, trends=trends)


def indicator_sensitivity(x: np.ndarray, windows: tuple[int, ...],
                          bandwidths: tuple[int, ...], dt: float = 1.0
                          ) -> dict[tuple[int, int], float]:
    """Kendall tau of the variance trend over a grid of filter choices.

    Reported alongside every early-warning result: an indicator that only trends
    for one particular window and bandwidth is an artefact of that choice, not
    evidence of lost resilience.
    """
    out: dict[tuple[int, int], float] = {}
    for w in windows:
        for b in bandwidths:
            if w >= len(x):
                continue
            resid = detrend(np.asarray(x, float), b)
            _, var = rolling_variance(resid, w)
            good = np.isfinite(var)
            out[(w, b)] = (float(stats.kendalltau(np.arange(var.size)[good],
                                                  var[good])[0])
                           if good.sum() >= 3 else np.nan)
    return out
