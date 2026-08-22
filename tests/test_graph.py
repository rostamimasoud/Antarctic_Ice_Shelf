"""Tests for graph construction."""

from __future__ import annotations

import numpy as np
import pytest

from aisgnn.config import EDGE_FEATURES, NODE_FEATURES
from aisgnn.data.graph import (
    GraphArrays,
    build_graph,
    coarsen,
    edge_features,
    radius_edges,
)


def synthetic_cavity(nx: int = 12, ny: int = 10, dx: float = 5000.0, seed: int = 0):
    """A small rectangular cavity with smooth, physically ordered fields."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx, indexing="xy")

    mask = np.ones((ny, nx), bool)
    mask[0, :] = False                       # grounded edge

    draft = -400.0 - 200.0 * xs / xs.max()
    bed = draft - 300.0 - 100.0 * rng.random((ny, nx))
    wct = draft - bed

    fields = {
        "T": -1.8 + 1.5 * xs / xs.max(),
        "S": 34.4 + 0.3 * rng.random((ny, nx)),
        "thermal_driving": 0.2 + 1.0 * xs / xs.max(),
        "ice_draft": draft,
        "water_column": wct,
        "bed_depth": bed,
        "slope_ice": 1e-3 * rng.random((ny, nx)),
        "slope_bed": 2e-3 * rng.random((ny, nx)),
        "dist_gl": xs,
        "dist_front": xs.max() - xs,
        "coriolis": np.full((ny, nx), -1.4e-4),
        "entry_depth": bed - 50.0,
    }
    # Guard against the feature list and this fixture drifting apart: a missing
    # key would otherwise surface as an unrelated KeyError in every test.
    missing = set(NODE_FEATURES) - set(fields)
    assert not missing, f"fixture is missing node features: {sorted(missing)}"
    target = 0.5 + 4.0 * xs / xs.max()
    return fields, mask, xs, ys, target, dx


# --------------------------------------------------------------------------- #
# Neighbour search
# --------------------------------------------------------------------------- #

def test_radius_edges_respects_the_radius():
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    ei = radius_edges(pos, radius=2.0, max_degree=None)
    d = np.linalg.norm(pos[ei[0]] - pos[ei[1]], axis=1)
    assert (d < 2.0).all() and (d > 0).all()
    assert ei.shape[1] == 2                  # the close pair, both directions


def test_radius_edges_has_no_self_loops():
    pos = np.random.default_rng(0).random((60, 2)) * 100.0
    ei = radius_edges(pos, radius=30.0, max_degree=None)
    assert (ei[0] != ei[1]).all()


def test_radius_edges_is_symmetric():
    pos = np.random.default_rng(1).random((40, 2)) * 50.0
    ei = radius_edges(pos, radius=20.0, max_degree=None)
    forward = {(int(a), int(b)) for a, b in zip(*ei)}
    assert all((b, a) in forward for a, b in forward)


def test_degree_cap_is_enforced():
    pos = np.random.default_rng(2).random((200, 2)) * 10.0
    ei = radius_edges(pos, radius=20.0, max_degree=8)   # radius covers everything
    _, counts = np.unique(ei[1], return_counts=True)
    assert counts.max() <= 8


def test_degree_cap_keeps_the_nearest_neighbours():
    pos = np.array([[0.0, 0.0]] + [[float(i), 0.0] for i in range(1, 11)])
    ei = radius_edges(pos, radius=100.0, max_degree=3)
    incoming = ei[0][ei[1] == 0]
    d = np.linalg.norm(pos[incoming] - pos[0], axis=1)
    assert len(incoming) == 3
    assert set(np.round(d).astype(int)) == {1, 2, 3}


def test_too_few_nodes_gives_no_edges():
    assert radius_edges(np.zeros((1, 2)), radius=5.0).shape == (2, 0)


# --------------------------------------------------------------------------- #
# Edge features
# --------------------------------------------------------------------------- #

def test_edge_features_are_antisymmetric_where_expected():
    """Reversing an edge flips the differences and the bearing, not the distance."""
    pos = np.array([[0.0, 0.0], [3.0, 4.0]])
    bed = np.array([-800.0, -700.0])
    draft = np.array([-400.0, -450.0])
    wct = draft - bed

    fwd = edge_features(np.array([[0], [1]]), pos, bed, draft, wct)[0]
    rev = edge_features(np.array([[1], [0]]), pos, bed, draft, wct)[0]
    names = list(EDGE_FEATURES)

    assert fwd[names.index("distance")] == pytest.approx(5.0)
    assert rev[names.index("distance")] == pytest.approx(5.0)
    for f in ("d_bed", "d_draft", "bearing_sin", "bearing_cos"):
        assert fwd[names.index(f)] == pytest.approx(-rev[names.index(f)])


def test_edge_feature_column_count_matches_names():
    pos = np.random.default_rng(3).random((20, 2)) * 100.0
    ei = radius_edges(pos, radius=50.0)
    e = edge_features(ei, pos, pos[:, 0], pos[:, 1], pos[:, 0] - pos[:, 1])
    assert e.shape == (ei.shape[1], len(EDGE_FEATURES))


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def test_build_graph_shapes_and_provenance():
    fields, mask, xs, ys, target, dx = synthetic_cavity()
    g = build_graph(fields, mask, xs, ys, target, radius=1.5 * dx,
                    cell_area=dx * dx, shelf="Test", scenario="present_day",
                    simulation="OPM021")

    assert g.n_nodes == int(mask.sum())
    assert g.x.shape == (g.n_nodes, len(NODE_FEATURES))
    assert g.edge_attr.shape == (g.n_edges, len(EDGE_FEATURES))
    assert g.y.shape == (g.n_nodes,)
    assert g.pos.shape == (g.n_nodes, 2)
    assert g.node_features == tuple(NODE_FEATURES)
    assert g.shelf == "Test" and g.simulation == "OPM021"
    assert g.grid_index.shape == (g.n_nodes, 2)


def test_build_graph_excludes_masked_cells():
    fields, mask, xs, ys, target, dx = synthetic_cavity()
    g = build_graph(fields, mask, xs, ys, target, radius=1.5 * dx, cell_area=dx * dx)
    # Row 0 is grounded and must contribute no nodes.
    assert (g.grid_index[:, 0] > 0).all()


def test_build_graph_rejects_missing_features():
    fields, mask, xs, ys, target, dx = synthetic_cavity()
    del fields["thermal_driving"]
    with pytest.raises(KeyError, match="thermal_driving"):
        build_graph(fields, mask, xs, ys, target, radius=dx, cell_area=dx * dx)


def test_build_graph_rejects_nan_features():
    """A NaN would propagate through message passing and train to a constant."""
    fields, mask, xs, ys, target, dx = synthetic_cavity()
    fields["T"] = fields["T"].copy()
    fields["T"][3, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_graph(fields, mask, xs, ys, target, radius=1.5 * dx, cell_area=dx * dx)


def test_build_graph_rejects_empty_mask():
    fields, mask, xs, ys, target, dx = synthetic_cavity()
    with pytest.raises(ValueError, match="no cells"):
        build_graph(fields, np.zeros_like(mask), xs, ys, target,
                    radius=dx, cell_area=dx * dx)


def test_larger_radius_gives_more_edges():
    fields, mask, xs, ys, target, dx = synthetic_cavity()
    small = build_graph(fields, mask, xs, ys, target, radius=1.2 * dx,
                        cell_area=dx * dx, max_degree=None)
    large = build_graph(fields, mask, xs, ys, target, radius=2.5 * dx,
                        cell_area=dx * dx, max_degree=None)
    assert large.n_edges > small.n_edges


def test_round_trip_through_disk(tmp_path):
    fields, mask, xs, ys, target, dx = synthetic_cavity()
    g = build_graph(fields, mask, xs, ys, target, radius=1.5 * dx,
                    cell_area=dx * dx, shelf="Test", scenario="4xCO2",
                    simulation="bi646")
    path = tmp_path / "g.npz"
    g.save(path)
    back = GraphArrays.load(path)

    assert np.allclose(back.x, g.x)
    assert np.array_equal(back.edge_index, g.edge_index)
    assert np.allclose(back.edge_attr, g.edge_attr)
    assert np.allclose(back.y, g.y)
    assert back.shelf == "Test" and back.scenario == "4xCO2"
    assert back.node_features == g.node_features
    back.validate()


def test_validate_catches_index_out_of_range():
    fields, mask, xs, ys, target, dx = synthetic_cavity()
    g = build_graph(fields, mask, xs, ys, target, radius=1.5 * dx, cell_area=dx * dx)
    g.edge_index = g.edge_index.copy()
    g.edge_index[0, 0] = g.n_nodes + 5
    with pytest.raises(ValueError, match="out of range"):
        g.validate()


def test_coarsen_reduces_node_count():
    fields, mask, xs, ys, target, dx = synthetic_cavity(nx=20, ny=20)
    fine = build_graph(fields, mask, xs, ys, target, radius=1.5 * dx, cell_area=dx * dx)
    coarse_mask = coarsen(mask, 2)
    coarse = build_graph(fields, coarse_mask, xs, ys, target, radius=3.0 * dx,
                         cell_area=4 * dx * dx)
    assert coarse.n_nodes < fine.n_nodes
    assert coarse.n_nodes == int(coarse_mask.sum())
