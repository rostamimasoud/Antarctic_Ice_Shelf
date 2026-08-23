"""Bifurcation analysis of a trained emulator.

The emulator is a *diagnostic* map: it predicts melt from an instantaneous ocean
state and does not evolve one.  Sweeping it therefore reconstructs the
quasi-static response curve, not a trajectory, and the two must not be conflated.
Concretely:

* A hysteresis loop cannot appear in a single forward or reverse sweep of a
  memoryless map -- the forward and reverse sweeps are identical by construction.
  What the sweep *can* find is a sharp, threshold-like steepening of the melt
  response, which is the emulator's signature of the underlying fold.
* To obtain genuine hysteresis the emulator must be closed into a feedback loop,
  which is what :func:`closed_loop_sweep` does: melt is fed back into the
  freshwater budget of the cavity, so the state carries memory between steps.

Both are provided, and the distinction is reported with the results rather than
buried, because a hysteresis width quoted from an open-loop sweep would be an
artefact of the plotting order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import NODE_FEATURES


@dataclass
class SweepResult:
    """Emulator response to a swept forcing perturbation."""

    parameter: str
    values: np.ndarray                     # forcing offsets applied
    melt: np.ndarray                       # area-weighted mean melt, m/yr
    melt_members: np.ndarray               # (n_members, n_steps) when ensembled
    gradient: np.ndarray                   # d(melt)/d(forcing)
    shelf: str = ""
    mode: str = "open_loop"
    embeddings: np.ndarray | None = None   # (n_steps, n_hidden) mean node embedding
    note: str = ""

    @property
    def steepest(self) -> int:
        return int(np.nanargmax(np.abs(self.gradient))) if self.gradient.size else -1


@dataclass
class ThresholdEstimate:
    """A threshold-like steepening in the emulator response."""

    parameter: str
    shelf: str
    location: float                        # forcing offset of maximum gradient
    max_gradient: float                    # m/yr per unit forcing
    sharpness: float                       # max gradient / median gradient
    melt_before: float
    melt_after: float
    is_abrupt: bool                        # sharpness above the detection cut
    members_abrupt: float = float("nan")   # fraction of ensemble members agreeing


#: A response is called abrupt when its steepest gradient exceeds this multiple
#: of the median gradient.  A smoothly saturating curve sits near 2-3.
SHARPNESS_CUT = 6.0


# --------------------------------------------------------------------------- #
# Perturbing the input state
# --------------------------------------------------------------------------- #

def perturb(data, parameter: str, offset: float, feature_names=NODE_FEATURES):
    """Return a copy of ``data`` with one input feature shifted.

    Thermal driving and temperature are shifted together when either is named,
    because they are not independent: raising the temperature at the ice draft
    raises the thermal driving by the same amount at fixed salinity and depth.
    Shifting only one produces a physically impossible state that the emulator
    has never seen, and the resulting response curve says more about
    extrapolation than about the ocean.
    """
    names = list(feature_names)
    if parameter not in names:
        raise ValueError(f"{parameter!r} not among {names}")

    out = data.clone()
    x = out.x.clone()
    x[:, names.index(parameter)] += offset

    if parameter in ("T", "thermal_driving"):
        partner = "thermal_driving" if parameter == "T" else "T"
        if partner in names:
            x[:, names.index(partner)] += offset

    out.x = x
    return out


def area_weights(data) -> torch.Tensor:
    """Normalised cell areas, falling back to uniform weights."""
    area = getattr(data, "area", None)
    if area is None:
        return torch.full_like(data.y, 1.0 / data.y.numel())
    return area / area.sum()


# --------------------------------------------------------------------------- #
# Open-loop sweep
# --------------------------------------------------------------------------- #

@torch.no_grad()
def sweep(models, data, parameter: str = "thermal_driving",
          offsets: np.ndarray | None = None,
          collect_embeddings: bool = False) -> SweepResult:
    """Sweep a forcing offset through one or more trained emulators.

    ``models`` may be a single model or a list forming an ensemble; members are
    swept independently and reported separately, because averaging predictions
    before locating a threshold smooths a sharp response into a gradual one.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    if offsets is None:
        offsets = np.linspace(-1.0, 4.0, 101)

    weights = area_weights(data)
    members = np.empty((len(models), offsets.size))
    embeddings = [] if collect_embeddings else None

    for j, model in enumerate(models):
        model.eval()
        for i, off in enumerate(offsets):
            pred = model.denormalise(model(perturb(data, parameter, float(off))))
            members[j, i] = float((pred * weights).sum())
            if collect_embeddings and j == 0:
                emb = model.node_embeddings
                embeddings.append(emb.mean(dim=0).cpu().numpy()
                                  if emb is not None else np.zeros(1))

    melt = members.mean(axis=0)
    gradient = np.gradient(melt, offsets)

    return SweepResult(
        parameter=parameter, values=np.asarray(offsets, float), melt=melt,
        melt_members=members, gradient=gradient,
        shelf=getattr(data, "shelf", ""),
        mode="open_loop",
        embeddings=np.asarray(embeddings) if collect_embeddings else None,
        note="diagnostic map: forward and reverse sweeps coincide by construction",
    )


def detect_threshold(result: SweepResult,
                     sharpness_cut: float = SHARPNESS_CUT) -> ThresholdEstimate:
    """Locate the steepest part of a swept response and judge whether it is abrupt."""
    grad = np.abs(result.gradient)
    finite = grad[np.isfinite(grad)]
    median = float(np.median(finite)) if finite.size else np.nan
    i = result.steepest

    if i < 0 or not np.isfinite(median) or median <= 0:
        return ThresholdEstimate(result.parameter, result.shelf, np.nan, np.nan,
                                 np.nan, np.nan, np.nan, False)

    sharpness = float(grad[i] / median)
    lo = max(i - 5, 0)
    hi = min(i + 6, result.melt.size - 1)

    members_abrupt = np.nan
    if result.melt_members.shape[0] > 1:
        flags = []
        for row in result.melt_members:
            g = np.abs(np.gradient(row, result.values))
            med = np.median(g[np.isfinite(g)])
            flags.append(bool(med > 0 and np.nanmax(g) / med > sharpness_cut))
        members_abrupt = float(np.mean(flags))

    return ThresholdEstimate(
        parameter=result.parameter, shelf=result.shelf,
        location=float(result.values[i]), max_gradient=float(grad[i]),
        sharpness=sharpness, melt_before=float(result.melt[lo]),
        melt_after=float(result.melt[hi]),
        is_abrupt=sharpness > sharpness_cut, members_abrupt=members_abrupt)


# --------------------------------------------------------------------------- #
# Closed-loop sweep
# --------------------------------------------------------------------------- #

@dataclass
class HysteresisResult:
    """Forward and reverse branches of a closed-loop sweep."""

    parameter: str
    shelf: str
    values: np.ndarray
    forward: np.ndarray
    reverse: np.ndarray
    width: float                           # max separation between branches
    loop_area: float                       # enclosed area, the hysteresis measure
    note: str = ""
    history: dict = field(default_factory=dict)


@torch.no_grad()
def closed_loop_sweep(model, data, parameter: str = "thermal_driving",
                      offsets: np.ndarray | None = None,
                      feedback: float = 0.15, relax: float = 0.35,
                      n_relax: int = 40) -> HysteresisResult:
    """Sweep with meltwater fed back into the cavity state.

    The emulator alone has no memory, so hysteresis requires closing the loop
    that produces it physically: meltwater freshens the cavity, which lowers the
    thermal driving the emulator then sees.  At each forcing step the state is
    relaxed to a fixed point of

        ``thermal_driving = base + offset - feedback * melt(thermal_driving)``

    and the forward and reverse branches start from opposite ends, so a genuine
    bistability shows up as the two branches settling on different fixed points
    over the same forcing range.

    ``feedback`` sets the strength of the meltwater damping in degC per m/yr of
    melt; it is a free parameter of this diagnostic and is varied in the
    sensitivity analysis rather than treated as calibrated.
    """
    if offsets is None:
        offsets = np.linspace(-1.0, 4.0, 81)
    offsets = np.asarray(offsets, float)

    names = list(NODE_FEATURES)
    idx = names.index(parameter)
    weights = area_weights(data)
    model.eval()

    def settle(offset: float, start: float) -> tuple[float, float]:
        state = start
        for _ in range(n_relax):
            work = data.clone()
            x = work.x.clone()
            x[:, idx] = x[:, idx] + offset - state
            if parameter in ("T", "thermal_driving"):
                partner = "thermal_driving" if parameter == "T" else "T"
                if partner in names:
                    j = names.index(partner)
                    x[:, j] = x[:, j] + offset - state
            work.x = x
            melt = float((model.denormalise(model(work)) * weights).sum())
            target = feedback * melt
            new = state + relax * (target - state)
            if abs(new - state) < 1e-6:
                state = new
                break
            state = new
        return melt, state

    forward = np.empty(offsets.size)
    state = 0.0
    for i, off in enumerate(offsets):
        forward[i], state = settle(float(off), state)

    reverse = np.empty(offsets.size)
    for i, off in enumerate(offsets[::-1]):
        reverse[offsets.size - 1 - i], state = settle(float(off), state)

    gap = np.abs(forward - reverse)
    return HysteresisResult(
        parameter=parameter, shelf=getattr(data, "shelf", ""),
        values=offsets, forward=forward, reverse=reverse,
        width=float(np.nanmax(gap)),
        loop_area=float(np.trapz(gap, offsets)),
        note=f"closed loop, feedback={feedback} degC per m/yr",
    )


# --------------------------------------------------------------------------- #
# Phase space
# --------------------------------------------------------------------------- #

def phase_space(embeddings: np.ndarray, n_components: int = 2
                ) -> tuple[np.ndarray, np.ndarray]:
    """Project swept hidden states onto their leading principal components.

    Returns ``(scores, explained_variance_ratio)``.  A fold appears as a sharp
    turn or jump in the trajectory rather than as a smooth arc.
    """
    x = np.asarray(embeddings, float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"need a (n_steps, n_hidden) array, got {x.shape}")

    centred = x - x.mean(axis=0)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    k = min(n_components, s.size)
    scores = u[:, :k] * s[:k]
    total = float((s ** 2).sum())
    ratio = (s[:k] ** 2) / total if total > 0 else np.zeros(k)
    return scores, ratio
