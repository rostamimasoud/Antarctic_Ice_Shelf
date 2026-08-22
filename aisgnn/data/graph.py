"""Graph construction from gridded cavity fields.

Each ice-shelf cavity becomes a graph whose nodes are the wet grid cells beneath
the shelf.  Edges connect cells within a fixed radius and carry the geometric
information a node-only model cannot see: separation, bathymetric and ice-draft
gradients, orientation, and alignment with the local water-column-thickness
contour, which is the direction a rotating, topographically steered inflow tends
to follow.

The construction is deliberately plain NumPy and returns
:class:`GraphArrays`; conversion to a PyTorch Geometric ``Data`` object is a
separate step.  That keeps the geometry testable without a GPU stack, and keeps
the expensive neighbour search independent of the deep-learning framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EDGE_FEATURES, NODE_FEATURES


@dataclass
class GraphArrays:
    """A cavity graph in plain arrays.

    Attributes
    ----------
    x
        Node features, ``(n_nodes, n_node_features)``.
    edge_index
        Source and target node indices, ``(2, n_edges)``, in the PyTorch
        Geometric convention where ``edge_index[0]`` sends to ``edge_index[1]``.
    edge_attr
        Edge features, ``(n_edges, n_edge_features)``.
    y
        Target melt rate per node, ``(n_nodes,)``, in m/yr.
    pos
        Node coordinates in projected metres, ``(n_nodes, 2)``.
    area
        Cell area per node in m2, used for area-weighted aggregation.
    node_features, edge_features
        Names, in column order.
    shelf, scenario, simulation
        Provenance.
    grid_index
        Row and column of each node in the source grid, ``(n_nodes, 2)``, so
        predictions can be mapped back onto the grid for plotting.
    """

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    y: np.ndarray
    pos: np.ndarray
    area: np.ndarray
    node_features: tuple[str, ...]
    edge_features: tuple[str, ...]
    shelf: str = ""
    scenario: str = ""
    simulation: str = ""
    grid_index: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def summary(self) -> str:
        deg = self.n_edges / max(self.n_nodes, 1)
        return (f"{self.shelf} [{self.simulation}/{self.scenario}]: "
                f"{self.n_nodes} nodes, {self.n_edges} edges, mean degree {deg:.1f}, "
                f"melt {np.nanmin(self.y):.2f} to {np.nanmax(self.y):.2f} m/yr")

    def validate(self) -> None:
        """Raise if the graph is internally inconsistent or carries NaNs.

        Called before every graph is written.  A NaN in a node feature
        propagates silently through message passing and produces a model that
        trains to a constant, which is easy to mistake for a physical result.
        """
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"{self.shelf}: {self.x.shape[0]} nodes but "
                             f"{self.y.shape[0]} targets")
        if self.x.shape[1] != len(self.node_features):
            raise ValueError(f"{self.shelf}: {self.x.shape[1]} feature columns but "
                             f"{len(self.node_features)} names")
        if self.edge_index.shape[0] != 2:
            raise ValueError(f"{self.shelf}: edge_index must have shape (2, n_edges)")
        if self.edge_attr.shape[0] != self.edge_index.shape[1]:
            raise ValueError(f"{self.shelf}: {self.edge_attr.shape[0]} edge features "
                             f"for {self.edge_index.shape[1]} edges")
        if self.n_edges and int(self.edge_index.max()) >= self.n_nodes:
            raise ValueError(f"{self.shelf}: edge index out of range")
        if not np.isfinite(self.x).all():
            bad = [n for i, n in enumerate(self.node_features)
                   if not np.isfinite(self.x[:, i]).all()]
            raise ValueError(f"{self.shelf}: non-finite node features: {bad}")
        if not np.isfinite(self.edge_attr).all():
            raise ValueError(f"{self.shelf}: non-finite edge features")
        if not np.isfinite(self.y).all():
            raise ValueError(f"{self.shelf}: non-finite targets")

    # -- persistence -------------------------------------------------------- #

    def save(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, x=self.x, edge_index=self.edge_index, edge_attr=self.edge_attr,
            y=self.y, pos=self.pos, area=self.area,
            grid_index=(self.grid_index if self.grid_index is not None
                        else np.empty((0, 2), int)),
            node_features=np.array(self.node_features),
            edge_features=np.array(self.edge_features),
            meta=np.array([self.shelf, self.scenario, self.simulation]))

    @classmethod
    def load(cls, path) -> "GraphArrays":
        blob = np.load(path, allow_pickle=False)
        shelf, scenario, simulation = (str(v) for v in blob["meta"])
        gi = blob["grid_index"]
        return cls(x=blob["x"], edge_index=blob["edge_index"],
                   edge_attr=blob["edge_attr"], y=blob["y"], pos=blob["pos"],
                   area=blob["area"],
                   node_features=tuple(str(v) for v in blob["node_features"]),
                   edge_features=tuple(str(v) for v in blob["edge_features"]),
                   shelf=shelf, scenario=scenario, simulation=simulation,
                   grid_index=gi if gi.size else None)

    def to_pyg(self):
        """Convert to a PyTorch Geometric ``Data`` object."""
        import torch
        from torch_geometric.data import Data

        data = Data(
            x=torch.as_tensor(self.x, dtype=torch.float32),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.as_tensor(self.edge_attr, dtype=torch.float32),
            y=torch.as_tensor(self.y, dtype=torch.float32),
            pos=torch.as_tensor(self.pos, dtype=torch.float32),
        )
        data.area = torch.as_tensor(self.area, dtype=torch.float32)
        data.shelf = self.shelf
        data.scenario = self.scenario
        data.simulation = self.simulation
        return data


# --------------------------------------------------------------------------- #
# Neighbour search
# --------------------------------------------------------------------------- #

def radius_edges(pos: np.ndarray, radius: float, max_degree: int | None = 16
                 ) -> np.ndarray:
    """Directed edges between all node pairs closer than ``radius`` metres.

    A KD-tree is used where SciPy is available and a blocked brute-force search
    otherwise.  Self-loops are excluded; the convolutions add their own where
    they need them.

    Parameters
    ----------
    max_degree
        Cap on incoming edges per node, keeping the nearest.  Cavity grids are
        near-uniform so the degree is usually well below this, but the cap bounds
        memory when a coarsened grid leaves a few densely packed cells.
    """
    n = pos.shape[0]
    if n < 2:
        return np.empty((2, 0), dtype=np.int64)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pos)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if pairs.size == 0:
            return np.empty((2, 0), dtype=np.int64)
        # query_pairs returns each undirected pair once; make both directions.
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    except ImportError:
        src_list, dst_list = [], []
        block = 2048
        for start in range(0, n, block):
            stop = min(start + block, n)
            d = np.linalg.norm(pos[start:stop, None, :] - pos[None, :, :], axis=-1)
            i, j = np.nonzero((d < radius) & (d > 0))
            src_list.append(j)
            dst_list.append(i + start)
        src = np.concatenate(src_list) if src_list else np.empty(0, int)
        dst = np.concatenate(dst_list) if dst_list else np.empty(0, int)

    edge_index = np.vstack([src, dst]).astype(np.int64)

    if max_degree is not None and edge_index.shape[1]:
        edge_index = _cap_degree(edge_index, pos, max_degree)
    return edge_index


def _cap_degree(edge_index: np.ndarray, pos: np.ndarray, max_degree: int
                ) -> np.ndarray:
    """Keep only the ``max_degree`` nearest incoming edges per target node."""
    src, dst = edge_index
    dist = np.linalg.norm(pos[src] - pos[dst], axis=1)

    order = np.lexsort((dist, dst))
    dst_sorted = dst[order]
    # Rank of each edge within its target node's group.
    starts = np.searchsorted(dst_sorted, dst_sorted, side="left")
    rank = np.arange(dst_sorted.size) - starts
    keep = order[rank < max_degree]
    return edge_index[:, np.sort(keep)]


# --------------------------------------------------------------------------- #
# Edge features
# --------------------------------------------------------------------------- #

def edge_features(edge_index: np.ndarray, pos: np.ndarray, bed: np.ndarray,
                  draft: np.ndarray, water_column: np.ndarray) -> np.ndarray:
    """Geometric features for each edge, in the order of :data:`EDGE_FEATURES`.

    ``along_contour`` is the cosine of the angle between the edge and the local
    water-column-thickness contour.  Under rotation, sub-shelf flow tends to
    follow contours of water-column thickness rather than the steepest gradient,
    so this distinguishes edges along the likely flow path from edges across it.
    """
    src, dst = edge_index
    delta = pos[dst] - pos[src]
    dist = np.linalg.norm(delta, axis=1)
    safe = np.where(dist > 0, dist, 1.0)

    bearing_cos = delta[:, 0] / safe
    bearing_sin = delta[:, 1] / safe

    d_bed = bed[dst] - bed[src]
    d_draft = draft[dst] - draft[src]

    # Gradient of water-column thickness, estimated per edge; the contour
    # direction is perpendicular to it.
    d_wct = water_column[dst] - water_column[src]
    grad = d_wct / safe
    scale = np.percentile(np.abs(grad), 95) if grad.size else 1.0
    along_contour = 1.0 - np.abs(grad) / (scale if scale > 0 else 1.0)
    along_contour = np.clip(along_contour, -1.0, 1.0)

    return np.column_stack([dist, d_bed, d_draft, bearing_sin, bearing_cos,
                            along_contour]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def build_graph(fields: dict[str, np.ndarray], mask: np.ndarray,
                x_coord: np.ndarray, y_coord: np.ndarray,
                target: np.ndarray, radius: float,
                cell_area: np.ndarray | float,
                shelf: str = "", scenario: str = "", simulation: str = "",
                max_degree: int | None = 16,
                node_features: tuple[str, ...] = NODE_FEATURES) -> GraphArrays:
    """Assemble a cavity graph from gridded fields.

    Parameters
    ----------
    fields
        Mapping from feature name to a 2-D array on the model grid.  Every name
        in ``node_features`` must be present.
    mask
        Boolean grid, ``True`` for cavity cells to include.
    x_coord, y_coord
        Projected coordinates in metres, same shape as ``mask``.
    target
        Basal melt rate on the grid, in m/yr.
    radius
        Edge radius in metres.
    cell_area
        Cell area in m2, either a grid or a scalar.

    Returns
    -------
    GraphArrays
        A validated graph.
    """
    missing = [f for f in node_features if f not in fields]
    if missing:
        raise KeyError(f"{shelf}: missing node features {missing}")

    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError(f"{shelf}: mask selects no cells")

    sel = np.nonzero(mask)
    pos = np.column_stack([x_coord[sel], y_coord[sel]]).astype(np.float64)

    x = np.column_stack([np.asarray(fields[f])[sel] for f in node_features])
    x = x.astype(np.float32)
    y = np.asarray(target)[sel].astype(np.float32)

    area = (np.full(pos.shape[0], float(cell_area), dtype=np.float32)
            if np.isscalar(cell_area)
            else np.asarray(cell_area)[sel].astype(np.float32))

    edge_index = radius_edges(pos, radius, max_degree=max_degree)
    e_attr = edge_features(
        edge_index, pos,
        bed=np.asarray(fields["bed_depth"])[sel].astype(np.float64),
        draft=np.asarray(fields["ice_draft"])[sel].astype(np.float64),
        water_column=np.asarray(fields["water_column"])[sel].astype(np.float64),
    )

    graph = GraphArrays(
        x=x, edge_index=edge_index, edge_attr=e_attr, y=y, pos=pos.astype(np.float32),
        area=area, node_features=tuple(node_features),
        edge_features=tuple(EDGE_FEATURES), shelf=shelf, scenario=scenario,
        simulation=simulation,
        grid_index=np.column_stack(sel).astype(np.int32),
    )
    graph.validate()
    return graph


def coarsen(mask: np.ndarray, factor: int) -> np.ndarray:
    """Thin a mask by keeping every ``factor``-th row and column.

    Used for the multi-resolution training strategy: a graph at 4 km trains far
    faster than one at 2 km and is used to fix hyperparameters before the final
    fit at full resolution.
    """
    if factor < 1:
        raise ValueError("factor must be at least 1")
    out = np.zeros_like(mask, dtype=bool)
    out[::factor, ::factor] = mask[::factor, ::factor]
    return out
