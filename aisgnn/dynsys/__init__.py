"""Dynamical systems analysis: sweeps, bifurcation detection, early-warning signals."""

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

__all__ = [
    "EWSResult", "TrendTest", "compute_ews", "detrend", "indicator_sensitivity",
    "phase_randomised_surrogate", "rolling_autocorrelation",
    "rolling_irreversibility", "rolling_recovery_rate", "rolling_skewness",
    "rolling_variance", "time_irreversibility", "trend_significance",
]
