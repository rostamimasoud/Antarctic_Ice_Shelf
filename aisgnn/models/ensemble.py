"""Deep ensembles for uncertainty quantification.

Several independently initialised networks are trained on the same data and
their spread is used as the emulator's uncertainty.  This matters more here than
in a typical regression problem: the bifurcation diagrams in
:mod:`aisgnn.dynsys` are read off the emulator's response to a swept control
parameter, and a tipping threshold quoted without a confidence interval cannot
be compared against the box model in any meaningful way.

The ensemble deliberately does *not* average before detecting folds.  Averaging
predictions across members smooths a sharp transition into a gradual one and
would systematically understate how abrupt the shift is; instead each member is
swept separately and the resulting thresholds are pooled.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .architectures import ModelConfig, build_model


@dataclass
class EnsemblePrediction:
    """Per-node prediction statistics across ensemble members."""

    mean: np.ndarray
    std: np.ndarray
    members: np.ndarray          # (n_members, n_nodes)

    def interval(self, coverage: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Empirical prediction interval across members."""
        lo = (1.0 - coverage) / 2.0
        return (np.quantile(self.members, lo, axis=0),
                np.quantile(self.members, 1.0 - lo, axis=0))


class DeepEnsemble(nn.Module):
    """A set of independently initialised emulators of one architecture."""

    def __init__(self, arch: str, cfg: ModelConfig, n_members: int = 5,
                 seeds: Sequence[int] | None = None):
        super().__init__()
        self.arch = arch
        self.cfg = cfg
        self.seeds = list(seeds) if seeds is not None else list(range(n_members))

        members = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            members.append(build_model(arch, cfg))
        self.members = nn.ModuleList(members)

    def __len__(self) -> int:
        return len(self.members)

    # -- fitting the input/target scalings ---------------------------------- #

    @torch.no_grad()
    def fit_scalers(self, x: Tensor, y: Tensor) -> None:
        """Fit every member's standardiser on the same training statistics."""
        for m in self.members:
            m.standardiser.fit(x)
            m.fit_target(y)

    # -- inference ---------------------------------------------------------- #

    @torch.no_grad()
    def predict(self, data: Data, denormalise: bool = True) -> EnsemblePrediction:
        """Predict with every member and summarise the spread."""
        preds = []
        for m in self.members:
            m.eval()
            out = m(data)
            preds.append((m.denormalise(out) if denormalise else out).cpu().numpy())
        stack = np.stack(preds)
        return EnsemblePrediction(mean=stack.mean(0), std=stack.std(0, ddof=1)
                                  if len(preds) > 1 else np.zeros_like(stack[0]),
                                  members=stack)

    @torch.no_grad()
    def predict_aggregate(self, data: Data, weights: np.ndarray | None = None
                          ) -> np.ndarray:
        """Area-weighted mean melt per member, shape ``(n_members,)``.

        Used by the sweeps: each member yields its own bifurcation diagram.
        """
        pred = self.predict(data)
        if weights is None:
            return pred.members.mean(axis=1)
        w = np.asarray(weights, float)
        w = w / w.sum()
        return pred.members @ w

    def forward(self, data: Data) -> Tensor:
        """Mean prediction, for compatibility with single-model call sites."""
        return torch.stack([m(data) for m in self.members]).mean(dim=0)

    # -- persistence -------------------------------------------------------- #

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": self.arch, "cfg": vars(self.cfg), "seeds": self.seeds,
                    "state_dict": self.state_dict()}, path)
        return path

    @classmethod
    def load(cls, path: Path, map_location: str | torch.device = "cpu") -> "DeepEnsemble":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ModelConfig(**blob["cfg"])
        ens = cls(blob["arch"], cfg, seeds=blob["seeds"])
        ens.load_state_dict(blob["state_dict"])
        return ens


# --------------------------------------------------------------------------- #
# Pooling thresholds across members
# --------------------------------------------------------------------------- #

def pool_thresholds(values: Iterable[float], coverage: float = 0.9
                    ) -> dict[str, float]:
    """Summarise a per-member set of tipping thresholds.

    Members that did not tip contribute ``nan`` and are excluded from the
    location estimate but counted, because the fraction of members that tip is
    itself part of the result.
    """
    arr = np.asarray(list(values), float)
    finite = arr[np.isfinite(arr)]
    lo = (1.0 - coverage) / 2.0

    if finite.size == 0:
        return {"median": np.nan, "lower": np.nan, "upper": np.nan,
                "fraction_tipped": 0.0, "n_members": int(arr.size)}

    return {
        "median": float(np.median(finite)),
        "lower": float(np.quantile(finite, lo)),
        "upper": float(np.quantile(finite, 1.0 - lo)),
        "fraction_tipped": float(finite.size / arr.size),
        "n_members": int(arr.size),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, coverage: float = 0.95,
                 seed: int | None = 0) -> tuple[float, float]:
    """Bootstrap confidence interval on the median of ``values``."""
    arr = np.asarray(values, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(arr, size=(n_boot, arr.size), replace=True), axis=1)
    lo = (1.0 - coverage) / 2.0
    return float(np.quantile(meds, lo)), float(np.quantile(meds, 1.0 - lo))
