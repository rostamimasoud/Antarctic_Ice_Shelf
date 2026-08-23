"""Dataset assembly and splitting for the emulators.

Graphs are stored one per (simulation, shelf, year).  How they are split into
training, validation and test sets is the part that decides what a reported
score actually means, so the three regimes are explicit:

``random``
    shuffle all graphs.  Optimistic: the same shelf in adjacent years is nearly
    the same graph, so a random split leaks the test set into training and
    reports a skill the emulator does not have.
``shelf``
    hold out entire ice shelves.  Answers "does this generalise to a cavity it
    has never seen", which is the question for circum-Antarctic application.
``year``
    hold out later years.  Answers "does this extrapolate in time", which is the
    question for projections and the one Burgard et al. found hardest.
``scenario``
    train on one climate scenario and test on the other.  The hardest test and
    the one that matters for the tipping-point analysis.

``shelf`` is the default because a random split is misleading here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import GRAPH_DIR
from .graph import GraphArrays

_NAME = re.compile(r"^(?P<sim>[A-Za-z0-9_]+?)_(?P<shelf>[A-Za-z_\-]+?)_(?P<year>\d{4})\.npz$")

SPLIT_MODES = ("shelf", "year", "scenario", "random")


@dataclass
class GraphRecord:
    """One graph on disk, with its provenance parsed from the filename."""

    path: Path
    simulation: str
    shelf: str
    year: int

    def load(self) -> GraphArrays:
        return GraphArrays.load(self.path)


@dataclass
class Split:
    """A train/validation/test partition."""

    mode: str
    train: list[GraphRecord] = field(default_factory=list)
    val: list[GraphRecord] = field(default_factory=list)
    test: list[GraphRecord] = field(default_factory=list)
    note: str = ""

    def summary(self) -> str:
        def describe(records):
            shelves = sorted({r.shelf for r in records})
            years = sorted({r.year for r in records})
            return (f"{len(records):4d} graphs, {len(shelves)} shelves, "
                    f"{len(years)} years")
        return (f"split={self.mode}  {self.note}\n"
                f"  train {describe(self.train)}\n"
                f"  val   {describe(self.val)}\n"
                f"  test  {describe(self.test)}")


def index_graphs(directory: Path | None = None,
                 simulations: tuple[str, ...] | None = None) -> list[GraphRecord]:
    """Index the graph files on disk."""
    directory = Path(directory) if directory is not None else GRAPH_DIR
    records: list[GraphRecord] = []
    for path in sorted(directory.glob("*.npz")):
        m = _NAME.match(path.name)
        if not m:
            continue
        sim = m.group("sim")
        if simulations and sim not in simulations:
            continue
        records.append(GraphRecord(path=path, simulation=sim,
                                   shelf=m.group("shelf").replace("_", " "),
                                   year=int(m.group("year"))))
    return records


def make_split(records: list[GraphRecord], mode: str = "shelf",
               val_fraction: float = 0.2, test_fraction: float = 0.2,
               seed: int = 0, holdout_shelves: tuple[str, ...] = (),
               test_scenario: str | None = None) -> Split:
    """Partition graphs according to one of :data:`SPLIT_MODES`."""
    if mode not in SPLIT_MODES:
        raise ValueError(f"mode must be one of {SPLIT_MODES}, got {mode!r}")
    if not records:
        raise ValueError("no graphs to split")

    rng = np.random.default_rng(seed)

    if mode == "shelf":
        shelves = sorted({r.shelf for r in records})
        if holdout_shelves:
            test_shelves = [s for s in shelves if s in holdout_shelves]
            rest = [s for s in shelves if s not in test_shelves]
        else:
            order = [str(v) for v in rng.permutation(shelves)]
            n_test = max(1, int(round(test_fraction * len(order))))
            test_shelves, rest = order[:n_test], order[n_test:]
        n_val = max(1, int(round(val_fraction * len(rest))))
        val_shelves, train_shelves = rest[:n_val], rest[n_val:]
        note = f"test shelves {sorted(test_shelves)}, val {sorted(val_shelves)}"
        return Split(mode, [r for r in records if r.shelf in train_shelves],
                     [r for r in records if r.shelf in val_shelves],
                     [r for r in records if r.shelf in test_shelves], note)

    if mode == "year":
        years = sorted({r.year for r in records})
        n_test = max(1, int(round(test_fraction * len(years))))
        n_val = max(1, int(round(val_fraction * len(years))))
        test_years = years[-n_test:]
        val_years = years[-(n_test + n_val):-n_test]
        train_years = years[:-(n_test + n_val)] or years[:1]
        note = f"train <= {max(train_years)}, test >= {min(test_years)}"
        return Split(mode, [r for r in records if r.year in train_years],
                     [r for r in records if r.year in val_years],
                     [r for r in records if r.year in test_years], note)

    if mode == "scenario":
        sims = sorted({r.simulation for r in records})
        if test_scenario is None:
            if len(sims) < 2:
                raise ValueError(f"scenario split needs two simulations, found {sims}")
            test_scenario = sims[-1]
        train_pool = [r for r in records if r.simulation != test_scenario]
        if not train_pool:
            raise ValueError(f"no graphs left after holding out {test_scenario}")
        shelves = sorted({r.shelf for r in train_pool})
        order = [str(v) for v in rng.permutation(shelves)]
        n_val = max(1, int(round(val_fraction * len(order))))
        val_shelves = order[:n_val]
        note = f"test scenario {test_scenario}"
        return Split(mode,
                     [r for r in train_pool if r.shelf not in val_shelves],
                     [r for r in train_pool if r.shelf in val_shelves],
                     [r for r in records if r.simulation == test_scenario], note)

    order = list(rng.permutation(len(records)))
    n_test = max(1, int(round(test_fraction * len(order))))
    n_val = max(1, int(round(val_fraction * len(order))))
    idx_test = order[:n_test]
    idx_val = order[n_test:n_test + n_val]
    idx_train = order[n_test + n_val:]
    return Split(mode, [records[i] for i in idx_train],
                 [records[i] for i in idx_val],
                 [records[i] for i in idx_test],
                 "random split: optimistic, adjacent years leak")


def load_batch(records: list[GraphRecord], device: str = "cpu") -> list:
    """Load a set of graphs as PyTorch Geometric ``Data`` objects."""
    return [r.load().to_pyg().to(device) for r in records]


def stack_features(records: list[GraphRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate node features and targets, for fitting the scalers."""
    xs, ys = [], []
    for record in records:
        g = record.load()
        xs.append(g.x)
        ys.append(g.y)
    return np.concatenate(xs), np.concatenate(ys)
