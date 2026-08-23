"""Interpretability: learned connectivity, attribution and intervention.

Every routine here drives a trained emulator and therefore needs torch, so the
imports are lazy: importing :mod:`aisgnn` must not require a GPU stack.
"""

_NAMES = {
    "Attribution": "intervention",
    "LengthScale": "attention",
    "attention_length_scale": "attention",
    "compare_scales": "attention",
    "control_shift": "intervention",
    "dominant_control": "intervention",
    "integrated_gradients": "intervention",
    "intervene": "intervention",
    "sensitivity_length_scale": "attention",
}

__all__ = sorted(_NAMES)


def __getattr__(name: str):
    module = _NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)
