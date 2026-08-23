"""Spatial connectivity from learned attention.

The quantity H1 and H4 turn on is the distance over which an upstream cell
influences melt at a downstream one.  Two routes to it are provided, and they
answer subtly different questions:

``attention_length_scale``
    reads the GAT's attention weights directly.  Cheap, but attention is a
    property of the architecture as much as of the physics: a network can route
    information through several layers, so single-layer attention understates the
    true receptive field, and attention mass is normalised per node so it always
    sums to one regardless of how far influence actually reaches.

``sensitivity_length_scale``
    perturbs an upstream node and measures the change in predicted melt
    downstream.  This is the physically meaningful definition -- it is what
    "influence" means -- and it applies to every architecture including the MLP
    baseline, which by construction must give a length scale of zero.  It is
    correspondingly more expensive.

The sensitivity route is the primary measure; attention is reported alongside it
as a cross-check.  Where the two disagree, the sensitivity is believed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


#: Influence below this fraction of the peak is treated as numerically zero.
#:
#: Perturbing one node changes distant predictions by a residue of order 1e-4 of
#: the near-field response, and that residue is flat with distance.  Including it
#: in the exponential fit is not a small error: the fit then sees a steep drop
#: followed by a long flat tail and returns an e-folding scale several times the
#: true one.
NOISE_FLOOR = 1.0e-2

#: A fit explaining less than this fraction of the variance is not reported as a
#: length scale, only as an upper bound.
MIN_R_SQUARED = 0.5

#: Fewer usable bins than this cannot constrain an exponential.
MIN_BINS = 4


@dataclass
class LengthScale:
    """Distance-resolved influence and the scale fitted to it.

    ``length_scale`` is NaN when the decay is unresolved -- either it happens
    inside the first bin, or the fit is too poor to trust.  ``upper_bound`` is
    then the distance beyond which influence has already fallen to the noise
    floor, which is the only defensible statement in that case.
    """

    shelf: str
    method: str
    distances: np.ndarray          # bin centres, m
    influence: np.ndarray          # mean influence in each bin, normalised
    length_scale: float            # e-folding distance, m; NaN if unresolved
    r_squared: float               # quality of the exponential fit
    max_distance: float            # largest separation sampled
    note: str = ""
    upper_bound: float = float("nan")   # m
    n_bins_used: int = 0

    @property
    def resolved(self) -> bool:
        return bool(np.isfinite(self.length_scale) and self.length_scale > 0)


def _exponential_fit(distance: np.ndarray, influence: np.ndarray
                     ) -> tuple[float, float, int]:
    """Fit ``influence ~ exp(-d / L)`` in log space, above the noise floor.

    Returns ``(L, r_squared, n_bins_used)``.  Bins at or below
    :data:`NOISE_FLOOR` times the peak are excluded, because they carry no
    signal and dominate the regression if kept.
    """
    finite = np.isfinite(influence) & np.isfinite(distance) & (influence > 0)
    if not finite.any():
        return float("nan"), float("nan"), 0

    peak = float(np.nanmax(influence[finite]))
    good = finite & (influence > NOISE_FLOOR * peak)
    if good.sum() < MIN_BINS:
        return float("nan"), float("nan"), int(good.sum())

    d = distance[good]
    y = np.log(influence[good])
    slope, intercept = np.polyfit(d, y, 1)
    if slope >= 0:
        return float("nan"), 0.0, int(good.sum())

    pred = slope * d + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(-1.0 / slope), r2, int(good.sum())


def _log_bins(distance: np.ndarray, n_bins: int) -> np.ndarray:
    """Logarithmically spaced bin edges spanning the sampled separations.

    Linear bins are unusable across this range of cavity sizes: Filchner-Ronne
    spans 700 km, so twelve linear bins are 60 km wide and the entire decay falls
    inside the first one.  Log spacing resolves the near field for large and
    small cavities alike.
    """
    positive = distance[distance > 0]
    if positive.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo = max(float(np.percentile(positive, 1)), 1.0)
    hi = float(np.percentile(positive, 95))
    if hi <= lo:
        hi = lo * 10.0
    return np.geomspace(lo, hi, n_bins + 1)


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #

@torch.no_grad()
def attention_length_scale(model, data, n_bins: int = 12) -> LengthScale:
    """Length scale implied by the GAT's attention weights.

    Attention is renormalised so that a node's self-edge does not dominate, and
    weights are pooled across layers and heads.
    """
    if not hasattr(model, "attention"):
        raise TypeError(f"{type(model).__name__} does not expose attention weights")

    model.eval()
    model(data, return_attention=True)
    layers = model.attention
    if not layers:
        raise RuntimeError("no attention captured; call with return_attention=True")

    pos = data.pos.cpu().numpy()
    dist_all, weight_all = [], []

    for edge_index, alpha in layers:
        ei = edge_index.cpu().numpy()
        a = alpha.mean(dim=1).cpu().numpy() if alpha.dim() > 1 else alpha.cpu().numpy()
        src, dst = ei[0], ei[1]
        keep = src != dst                       # drop self-loops the layer added
        d = np.linalg.norm(pos[src[keep]] - pos[dst[keep]], axis=1)
        dist_all.append(d)
        weight_all.append(a[keep])

    d = np.concatenate(dist_all)
    w = np.concatenate(weight_all)
    if d.size == 0:
        return LengthScale(getattr(data, "shelf", ""), "attention",
                           np.empty(0), np.empty(0), np.nan, np.nan, 0.0,
                           "no non-self edges")

    edges = _log_bins(d, n_bins)
    centres = np.sqrt(edges[:-1] * edges[1:])
    influence = np.array([w[(d >= lo) & (d < hi)].mean() if ((d >= lo) & (d < hi)).any()
                          else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    if np.nanmax(influence) > 0:
        influence = influence / np.nanmax(influence)

    L, r2, n_used = _exponential_fit(centres, influence)
    return LengthScale(getattr(data, "shelf", ""), "attention", centres, influence,
                       L, r2, float(d.max()),
                       "single-hop attention; understates the multi-layer receptive field",
                       n_bins_used=n_used)


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sensitivity_length_scale(model, data, feature: str = "thermal_driving",
                             feature_names=None, n_sources: int = 24,
                             delta: float = 0.5, n_bins: int = 12,
                             seed: int = 0) -> LengthScale:
    """Length scale from perturbing upstream cells and measuring downstream response.

    A random sample of source nodes is perturbed one at a time; the resulting
    absolute change in predicted melt is binned by distance from the source.
    This is the definition of influence the hypotheses are stated in, and unlike
    attention it is defined for every architecture.
    """
    from ..config import NODE_FEATURES

    names = list(feature_names or NODE_FEATURES)
    if feature not in names:
        raise ValueError(f"{feature!r} not among {names}")
    col = names.index(feature)

    model.eval()
    base = model.denormalise(model(data))
    pos = data.pos.cpu().numpy()
    n_nodes = pos.shape[0]

    rng = np.random.default_rng(seed)
    sources = rng.choice(n_nodes, size=min(n_sources, n_nodes), replace=False)

    dist_all, resp_all = [], []
    for s in sources:
        work = data.clone()
        x = work.x.clone()
        x[s, col] += delta
        work.x = x
        response = (model.denormalise(model(work)) - base).abs().cpu().numpy()
        d = np.linalg.norm(pos - pos[s], axis=1)
        mask = np.arange(n_nodes) != s          # exclude the perturbed node itself
        dist_all.append(d[mask])
        resp_all.append(response[mask])

    d = np.concatenate(dist_all)
    r = np.concatenate(resp_all)

    edges = _log_bins(d, n_bins)
    centres = np.sqrt(edges[:-1] * edges[1:])          # geometric bin centres
    influence = np.array([r[(d >= lo) & (d < hi)].mean() if ((d >= lo) & (d < hi)).any()
                          else np.nan for lo, hi in zip(edges[:-1], edges[1:])])

    shelf = getattr(data, "shelf", "")
    peak = np.nanmax(influence) if np.isfinite(influence).any() else 0.0
    if not peak > 0:
        # A model with no spatial coupling responds only at the perturbed node,
        # which is excluded, so every bin is zero.  That is the correct answer
        # for the MLP baseline and must be reported as zero, not as a failure.
        return LengthScale(shelf, "sensitivity", centres, influence, 0.0, np.nan,
                           float(d.max()),
                           "no downstream response: model has no spatial coupling",
                           upper_bound=0.0, n_bins_used=0)

    influence = influence / peak
    L, r2, n_used = _exponential_fit(centres, influence)

    # Distance beyond which the response has already reached the noise floor.
    above = np.flatnonzero(np.nan_to_num(influence) > NOISE_FLOOR)
    upper = float(centres[above[-1]]) if above.size else float(centres[0])

    if not np.isfinite(L) or not np.isfinite(r2) or r2 < MIN_R_SQUARED:
        reason = ("decay unresolved: falls to the noise floor within the first bins"
                  if n_used < MIN_BINS else
                  f"exponential fit poor (R2 = {r2:.2f}); reporting an upper bound only")
        return LengthScale(shelf, "sensitivity", centres, influence, float("nan"),
                           r2, float(d.max()), reason,
                           upper_bound=upper, n_bins_used=n_used)

    return LengthScale(shelf, "sensitivity", centres, influence, L, r2,
                       float(d.max()),
                       f"{len(sources)} source nodes, delta={delta}, "
                       f"{n_used} bins above the noise floor",
                       upper_bound=upper, n_bins_used=n_used)


def compare_scales(model, data, **kwargs) -> dict:
    """Both length scales for one model and graph, for cross-checking."""
    out = {"shelf": getattr(data, "shelf", ""),
           "sensitivity": sensitivity_length_scale(model, data, **kwargs)}
    try:
        out["attention"] = attention_length_scale(model, data)
    except (TypeError, RuntimeError):
        out["attention"] = None
    return out
