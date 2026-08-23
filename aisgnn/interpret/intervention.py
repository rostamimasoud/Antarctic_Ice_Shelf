"""Feature attribution and intervention analysis.

Two complementary answers to "what is the emulator using":

``integrated_gradients``
    attributes a prediction to its inputs along a path from a baseline.  Cheap
    and exact in the sense that attributions sum to the prediction difference,
    but it is a statement about the model's local gradient, not about the ocean.

``intervene``
    perturbs a feature and measures the change in predicted melt.  This is the
    interventional question -- what happens *if* thermal driving rises -- and is
    the one that supports a causal reading, subject to the standard caveat that
    the emulator can only be trusted where the training data constrained it.

Both are reported. Where a feature has high attribution but negligible
intervention response, the model is exploiting a correlate rather than a driver,
and that distinction is the point of computing both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import NODE_FEATURES


@dataclass
class Attribution:
    """Per-feature importance for one graph."""

    shelf: str
    scenario: str
    method: str
    features: tuple[str, ...]
    values: np.ndarray            # one score per feature
    normalised: np.ndarray        # scores scaled to sum to one

    def ranked(self) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.values))
        return [(self.features[i], float(self.values[i])) for i in order]


def _weights(data) -> torch.Tensor:
    area = getattr(data, "area", None)
    if area is None:
        return torch.full_like(data.y, 1.0 / data.y.numel())
    return area / area.sum()


# --------------------------------------------------------------------------- #
# Integrated gradients
# --------------------------------------------------------------------------- #

def integrated_gradients(model, data, baseline: torch.Tensor | None = None,
                         steps: int = 32,
                         feature_names=NODE_FEATURES) -> Attribution:
    """Attribute area-mean melt to each input feature.

    The baseline is the per-feature training mean rather than zeros: a
    zero-input ocean is not a physical state, and attributions measured against
    it describe the path out of an impossible configuration.
    """
    model.eval()
    x0 = data.x.detach()
    if baseline is None:
        baseline = model.standardiser.mean.detach().to(x0.device).expand_as(x0)

    weights = _weights(data)
    total = torch.zeros(x0.shape[1], device=x0.device)

    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=x0.device):
        work = data.clone()
        work.x = (baseline + alpha * (x0 - baseline)).requires_grad_(True)
        out = (model.denormalise(model(work)) * weights).sum()
        grad, = torch.autograd.grad(out, work.x)
        total += grad.mean(dim=0)

    attribution = (total / steps) * (x0 - baseline).mean(dim=0)
    values = attribution.detach().cpu().numpy()
    denom = np.abs(values).sum()
    return Attribution(getattr(data, "shelf", ""), getattr(data, "scenario", ""),
                       "integrated_gradients", tuple(feature_names), values,
                       values / denom if denom > 0 else values)


# --------------------------------------------------------------------------- #
# Intervention
# --------------------------------------------------------------------------- #

@torch.no_grad()
def intervene(model, data, deltas=(-0.2, -0.1, 0.1, 0.2),
              relative: bool = True,
              feature_names=NODE_FEATURES) -> dict[str, dict]:
    """Perturb each feature in turn and record the melt response.

    ``relative`` scales the perturbation by each feature's own spread, so that
    features with different units are compared on equal footing; otherwise
    ``deltas`` are absolute in the feature's units.
    """
    model.eval()
    names = list(feature_names)
    weights = _weights(data)
    base = float((model.denormalise(model(data)) * weights).sum())

    scale = (model.standardiser.scale.detach() if relative
             else torch.ones_like(model.standardiser.scale))

    out: dict[str, dict] = {}
    for j, name in enumerate(names):
        responses = {}
        for d in deltas:
            work = data.clone()
            x = work.x.clone()
            x[:, j] = x[:, j] + d * float(scale[j])
            work.x = x
            responses[d] = float((model.denormalise(model(work)) * weights).sum()) - base

        pos = [v for k, v in responses.items() if k > 0]
        neg = [v for k, v in responses.items() if k < 0]
        span = max(deltas) - min(deltas)
        out[name] = {
            "baseline_melt": base,
            "responses": responses,
            "sensitivity": (np.mean(pos) - np.mean(neg)) / span if pos and neg else np.nan,
            "asymmetry": (abs(np.mean(pos) + np.mean(neg)) if pos and neg else np.nan),
        }
    return out


def dominant_control(intervention: dict[str, dict]) -> tuple[str, float]:
    """The feature whose perturbation moves melt most."""
    ranked = sorted(intervention.items(),
                    key=lambda kv: -abs(kv[1].get("sensitivity", 0.0) or 0.0))
    name, record = ranked[0]
    return name, float(record["sensitivity"])


def control_shift(present: dict[str, dict], future: dict[str, dict]
                  ) -> dict[str, float]:
    """Change in each feature's interventional sensitivity between scenarios.

    This is the quantity the dominant-control hypothesis is about: not which
    feature matters most in either scenario alone, but which gains influence as
    the climate warms.
    """
    shared = set(present) & set(future)
    out = {}
    for name in sorted(shared):
        a = present[name].get("sensitivity", np.nan)
        b = future[name].get("sensitivity", np.nan)
        out[name] = float(b - a)
    return out
