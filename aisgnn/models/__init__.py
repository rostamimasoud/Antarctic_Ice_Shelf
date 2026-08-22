"""Emulator architectures and deep ensembles.

Torch is imported lazily so that the box-model and early-warning-signal code,
which needs neither torch nor PyTorch Geometric, remains importable in a plain
scientific-Python environment.
"""

__all__ = [
    "ARCHITECTURES", "DeepEnsemble", "EnsemblePrediction", "MeltEGCN", "MeltGAT",
    "MeltGCN", "MeltMLP", "ModelConfig", "bootstrap_ci", "build_model",
    "pool_thresholds",
]


def __getattr__(name: str):
    if name in {"ARCHITECTURES", "MeltEGCN", "MeltGAT", "MeltGCN", "MeltMLP",
                "ModelConfig", "build_model"}:
        from . import architectures
        return getattr(architectures, name)
    if name in {"DeepEnsemble", "EnsemblePrediction", "bootstrap_ci",
                "pool_thresholds"}:
        from . import ensemble
        return getattr(ensemble, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
