"""Emulator architectures.

Four models sharing one interface, so that the comparison between them isolates
the effect of spatial structure rather than of incidental differences in
training setup:

``MeltMLP``
    Per-cell multilayer perceptron with no spatial coupling, following the
    architecture of Burgard et al. (2023).  This is the baseline: it can only
    learn a local mapping from column properties to melt.
``MeltGCN``
    Graph convolution.  Neighbours are averaged with fixed, degree-normalised
    weights.
``MeltGAT``
    Graph attention (GATv2).  The attention weights are *learned* and are
    retained so that the length scale of upstream influence can be measured --
    this is the model H1 and H4 rely on.
``MeltEGCN``
    Edge-conditioned convolution, in which a hypernetwork turns edge features
    (separation, bathymetric and draft gradients, orientation) into the filter
    itself.  This is the architecture that can represent flow following
    bathymetric contours.

All four predict melt at every node and expose ``node_embeddings`` for the
phase-space analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, GCNConv, NNConv
from torch_geometric.utils import softmax as pyg_softmax


@dataclass
class ModelConfig:
    """Hyperparameters shared by every architecture."""

    in_channels: int
    edge_channels: int = 0
    hidden: int = 128
    layers: int = 4
    heads: int = 4                # GAT only
    dropout: float = 0.1
    activation: str = "gelu"
    residual: bool = True
    layer_norm: bool = True
    out_channels: int = 1
    extras: dict = field(default_factory=dict)


_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}


def _activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}; "
                         f"choose from {sorted(_ACTIVATIONS)}") from None


class _Standardiser(nn.Module):
    """Feature standardisation carried inside the model.

    Kept as buffers rather than applied in the data pipeline so that the
    statistics travel with the checkpoint.  A model loaded for a parameter sweep
    then cannot be silently fed unnormalised inputs, which would look like a
    physical result rather than a bug.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, x: Tensor, eps: float = 1e-6) -> None:
        self.mean.copy_(x.mean(dim=0))
        self.scale.copy_(x.std(dim=0).clamp_min(eps))
        self.fitted.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.scale


class _Base(nn.Module):
    """Shared encoder/decoder plumbing and target scaling."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.standardiser = _Standardiser(cfg.in_channels)
        self.encoder = nn.Sequential(
            nn.Linear(cfg.in_channels, cfg.hidden),
            _activation(cfg.activation),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden // 2),
            _activation(cfg.activation),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, cfg.out_channels),
        )
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_scale", torch.ones(1))
        self._embeddings: Tensor | None = None

    # -- target scaling ----------------------------------------------------- #

    @torch.no_grad()
    def fit_target(self, y: Tensor, eps: float = 1e-6) -> None:
        self.target_mean.fill_(float(y.mean()))
        self.target_scale.fill_(max(float(y.std()), eps))

    def denormalise(self, y: Tensor) -> Tensor:
        """Map a network output back to melt rate in m/yr."""
        return y * self.target_scale + self.target_mean

    # -- introspection ------------------------------------------------------ #

    @property
    def node_embeddings(self) -> Tensor | None:
        """Final hidden layer from the last forward pass, for phase-space work."""
        return self._embeddings

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #

class MeltMLP(_Base):
    """Per-cell baseline with no spatial coupling.

    Any skill the graph models gain over this is attributable to spatial
    information, which is the comparison the study rests on.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        blocks = []
        for _ in range(cfg.layers):
            blocks.append(nn.Sequential(
                nn.LayerNorm(cfg.hidden) if cfg.layer_norm else nn.Identity(),
                nn.Linear(cfg.hidden, cfg.hidden),
                _activation(cfg.activation),
                nn.Dropout(cfg.dropout),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, data: Data) -> Tensor:
        h = self.encoder(self.standardiser(data.x))
        for block in self.blocks:
            h = h + block(h) if self.cfg.residual else block(h)
        self._embeddings = h
        return self.decoder(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# Graph models
# --------------------------------------------------------------------------- #

class MeltGCN(_Base):
    """Degree-normalised graph convolution with fixed neighbour weights."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.convs = nn.ModuleList(
            GCNConv(cfg.hidden, cfg.hidden, add_self_loops=True)
            for _ in range(cfg.layers))
        self.norms = nn.ModuleList(
            (nn.LayerNorm(cfg.hidden) if cfg.layer_norm else nn.Identity())
            for _ in range(cfg.layers))
        self.act = _activation(cfg.activation)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, data: Data) -> Tensor:
        h = self.encoder(self.standardiser(data.x))
        ew = getattr(data, "edge_weight", None)
        for conv, norm in zip(self.convs, self.norms):
            out = self.drop(self.act(conv(norm(h), data.edge_index, ew)))
            h = h + out if self.cfg.residual else out
        self._embeddings = h
        return self.decoder(h).squeeze(-1)


class MeltGAT(_Base):
    """Graph attention network retaining its attention weights.

    ``GATv2`` is used rather than the original formulation because the latter's
    attention is *static*: the ranking of neighbours is independent of the query
    node, which would make the learned "connectivity" partly an artefact of the
    architecture rather than of the physics.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        if cfg.hidden % cfg.heads:
            raise ValueError(f"hidden ({cfg.hidden}) must be divisible by "
                             f"heads ({cfg.heads})")
        per_head = cfg.hidden // cfg.heads

        self.convs = nn.ModuleList(
            GATv2Conv(cfg.hidden, per_head, heads=cfg.heads, concat=True,
                      dropout=cfg.dropout,
                      edge_dim=cfg.edge_channels or None)
            for _ in range(cfg.layers))
        self.norms = nn.ModuleList(
            (nn.LayerNorm(cfg.hidden) if cfg.layer_norm else nn.Identity())
            for _ in range(cfg.layers))
        self.act = _activation(cfg.activation)
        self._attention: list[tuple[Tensor, Tensor]] = []

    def forward(self, data: Data, return_attention: bool = False) -> Tensor:
        h = self.encoder(self.standardiser(data.x))
        edge_attr = getattr(data, "edge_attr", None) if self.cfg.edge_channels else None
        self._attention = []

        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                out, (idx, alpha) = conv(norm(h), data.edge_index, edge_attr,
                                         return_attention_weights=True)
                self._attention.append((idx.detach(), alpha.detach()))
            else:
                out = conv(norm(h), data.edge_index, edge_attr)
            out = self.act(out)
            h = h + out if self.cfg.residual else out

        self._embeddings = h
        return self.decoder(h).squeeze(-1)

    @property
    def attention(self) -> list[tuple[Tensor, Tensor]]:
        """Per-layer ``(edge_index, alpha)`` from the last attention-returning pass.

        ``alpha`` has shape ``(n_edges, heads)`` and is already normalised over
        each node's incoming edges.
        """
        return self._attention


class MeltEGCN(_Base):
    """Edge-conditioned convolution.

    A hypernetwork maps each edge's features to the weight matrix applied along
    that edge, so separation, bathymetric gradient and orientation modulate the
    exchange directly rather than only through the node features.  The
    hypernetwork output is ``hidden x hidden``, so the hidden width is kept
    modest and a bottleneck is used.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        if not cfg.edge_channels:
            raise ValueError("MeltEGCN requires edge features; set edge_channels > 0")

        bottleneck = cfg.extras.get("bottleneck", 32)
        self.project_in = nn.Linear(cfg.hidden, bottleneck)
        self.project_out = nn.Linear(bottleneck, cfg.hidden)

        self.convs = nn.ModuleList()
        for _ in range(cfg.layers):
            hyper = nn.Sequential(
                nn.Linear(cfg.edge_channels, 32),
                _activation(cfg.activation),
                nn.Linear(32, bottleneck * bottleneck),
            )
            self.convs.append(NNConv(bottleneck, bottleneck, hyper, aggr="mean"))

        self.norms = nn.ModuleList(
            (nn.LayerNorm(bottleneck) if cfg.layer_norm else nn.Identity())
            for _ in range(cfg.layers))
        self.act = _activation(cfg.activation)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, data: Data) -> Tensor:
        h = self.encoder(self.standardiser(data.x))
        z = self.project_in(h)
        for conv, norm in zip(self.convs, self.norms):
            out = self.drop(self.act(conv(norm(z), data.edge_index, data.edge_attr)))
            z = z + out if self.cfg.residual else out
        h = self.project_out(z)
        self._embeddings = h
        return self.decoder(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

ARCHITECTURES = {"mlp": MeltMLP, "gcn": MeltGCN, "gat": MeltGAT, "egcn": MeltEGCN}


def build_model(arch: str, cfg: ModelConfig) -> _Base:
    """Instantiate an architecture by name."""
    try:
        return ARCHITECTURES[arch](cfg)
    except KeyError:
        raise ValueError(f"unknown architecture {arch!r}; "
                         f"choose from {sorted(ARCHITECTURES)}") from None


def attention_to_node_weights(edge_index: Tensor, alpha: Tensor,
                              num_nodes: int) -> Tensor:
    """Total attention each node receives, averaged over heads.

    Useful as a quick diagnostic; the distance-resolved analysis in
    :mod:`aisgnn.interpret.attention` is what H1 actually uses.
    """
    a = alpha.mean(dim=1) if alpha.dim() > 1 else alpha
    out = torch.zeros(num_nodes, device=alpha.device, dtype=a.dtype)
    return out.index_add_(0, edge_index[1], a)


def renormalise_attention(edge_index: Tensor, alpha: Tensor,
                          num_nodes: int) -> Tensor:
    """Renormalise attention over each target node's incoming edges."""
    a = alpha.mean(dim=1) if alpha.dim() > 1 else alpha
    return pyg_softmax(a, edge_index[1], num_nodes=num_nodes)
