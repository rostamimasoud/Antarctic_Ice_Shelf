"""Dynamical systems analysis: sweeps, bifurcation detection, early-warning signals.

The sweep machinery needs torch and PyTorch Geometric because it drives a trained
emulator; the early-warning-signal estimators need neither.  Sweeps are therefore
imported lazily, so that the box-model and EWS analysis stay usable in a plain
scientific-Python environment without a GPU stack installed.
"""

from .ews import (
    EWSResult,
    TrendTest,
    compute_ews,
    detrend,
    indicator_sensitivity,
    phase_randomised_surrogate,
    rolling_autocorrelation,
    rolling_irreversibility,
    rolling_recovery_rate,
    rolling_skewness,
    rolling_variance,
    time_irreversibility,
    trend_significance,
)

_SWEEP_NAMES = {
    "HysteresisResult", "SweepResult", "ThresholdEstimate", "closed_loop_sweep",
    "detect_threshold", "perturb", "phase_space", "sweep",
}

__all__ = [
    "EWSResult", "TrendTest", "compute_ews", "detrend", "indicator_sensitivity",
    "phase_randomised_surrogate", "rolling_autocorrelation",
    "rolling_irreversibility", "rolling_recovery_rate", "rolling_skewness",
    "rolling_variance", "time_irreversibility", "trend_significance",
    *sorted(_SWEEP_NAMES),
]


def __getattr__(name: str):
    if name in _SWEEP_NAMES:
        from . import sweeps
        return getattr(sweeps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
