"""Step 3 — relation-specific heterogeneous GNN encoder + attention fusion.

Architecture (per CLAUDE.md's "Proposed pipeline", step 3): every node type
gets a per-relation embedding (``RelationSpecificConv`` -- one learnable
weight matrix per edge type, no shared/collapsed weights across relations),
then a semantic-level attention (``SemanticAttention``) fuses a node type's
relation-specific embeddings into one node embedding. Both the per-relation
embeddings and the fused embedding are kept (``RelationSpecificOutput``) so a
later continual-learning component (relation-specific memory bank,
transferability estimation) can use either.

This module does not modify the frozen Step 1/2 pipeline or the on-disk
graphs (``data/graphs*/*.pt``) in any way -- it only reads the ``HeteroData``
mini-graphs those steps already produced, exactly as stored. As of
2026-07-19 the schema is **11 relations**: the original 6 one-directional
relations (``host--originates-->flow``, ``flow--terminates_at-->host``,
``flow--targets_port-->port``, ``flow--uses_protocol-->protocol``,
``flow--uses_service-->service``, ``host--communicates_with-->host``) plus 5
reverse relations (``flow--originated_by-->host``,
``host--terminated_by-->flow``, ``port--targeted_by-->flow``,
``protocol--protocol_of-->flow``, ``service--service_of-->flow``), added per
the professor's explicit instruction. Unlike the earlier rejected
``ToUndirected()`` reverse-edge attempt (see git history, 2026-07-17), these
5 are individually named with correctly passive-voice semantics (e.g. "this
flow was originated by this host") rather than a generic ``rev_X`` prefix, so
they don't blur relation-specific meaning the way a synthetic
``rev_originates`` would have.

One consequence, verified directly against a regenerated real mini-graph:
incoming-relation counts per node type are no longer as asymmetric as before.
Flow now has 5 incoming relations (``originates`` from Host, plus
``targeted_by``, ``protocol_of``, ``service_of``, ``terminated_by``); Host
has 3 (``terminates_at`` and ``originated_by`` from Flow, ``communicates_with``
from Host); Protocol, Service, and Port still have exactly one each
(``uses_protocol``, ``uses_service``, ``targets_port``, all from Flow) --
unaffected, since the new relations only enrich Flow's and Host's incoming
side. For those three, ``SemanticAttention`` fusion is still a no-op over a
single relation (beta=1.0); for Flow and Host it is now a genuine multi-
relation combination.

**Per-relation self-preservation** (professor's instruction, 2026-07-19):
richer incoming relations alone don't put a node's own features into its
embedding -- ``RelationSpecificConv`` (see that module's docstring) now
concatenates each destination node's own current embedding into every one of
its relation-specific aggregations before a per-relation combine layer
projects back to ``hidden_dim``. ``SemanticAttention`` fusion is unchanged
and needs no self-term: it only ever combines the relation embeddings it's
given, and by the time it runs those already carry self-information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch_geometric.data import HeteroData

from trench_ids.model.attention_fusion import SemanticAttention
from trench_ids.model.relation_conv import RelationSpecificConv

FLOW_FEATURE_DIM = 37  # len(configs/graph.yaml: features)
HOST_FEATURE_DIM = 4  # total_flows, avg_bytes_as_src, avg_bytes_as_dst, unique_ports_contacted


def port_bucket(port_number: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """Log1p-scaled bucket index in ``[0, num_buckets)`` for a raw destination port.

    Port node identity is chunk-local (``graphs.py``) and spans the full
    0-65535 range, so an ``nn.Embedding`` keyed directly on the raw port
    number would need a 65536-row table whose rows mostly never repeat
    across mini-graphs -- no generalization benefit, all parameter cost.
    Log-spaced bucketing keeps well-known ports (0-1023) relatively finely
    separated while coarsely grouping the sparse high end, and bounds the
    embedding table to ``num_buckets`` rows regardless of port range.
    """
    clipped = port_number.clamp(min=0, max=65535).float()
    scaled = torch.log1p(clipped) / math.log1p(65535)
    idx = (scaled * (num_buckets - 1)).round().long()
    return idx.clamp(0, num_buckets - 1)


class NodeFeatureEncoders(nn.Module):
    """Map each node type's raw Step 2 features into a shared ``hidden_dim``.

    Flow/Host raw features are NetFlow byte/packet counts on wildly
    different scales (e.g. ``IN_BYTES`` can be orders of magnitude larger
    than ``TCP_FLAGS``) and are not normalized upstream (Step 1/2 store raw
    values). A per-sample ``LayerNorm`` on the raw feature vector, before the
    linear projection, was added after an integration test on real saved
    mini-graphs showed loss gradients on the order of 1e12 without it --
    confirmed the fix by rerunning the same test post-change. This is purely
    a model-input concern (does not touch the frozen Step 1/2 pipeline or
    data on disk). Protocol/Service are indexed by the *global* vocabulary
    (``trench_ids.vocab``, stable across every task/graph_size/benign_ratio
    set) so the same raw value gets the same embedding row everywhere --
    required for any later cross-task comparison of these node types'
    embeddings. Port has no such global vocabulary (chunk-local, no cross-
    task comparability requirement per ``vocab.py``'s docstring), so it goes
    through ``port_bucket`` instead of a vocab lookup.
    """

    def __init__(
        self,
        hidden_dim: int,
        protocol_vocab_size: int,
        service_vocab_size: int,
        flow_feature_dim: int = FLOW_FEATURE_DIM,
        host_feature_dim: int = HOST_FEATURE_DIM,
        port_buckets: int = 32,
    ) -> None:
        super().__init__()
        self.flow_norm = nn.LayerNorm(flow_feature_dim)
        self.flow = nn.Linear(flow_feature_dim, hidden_dim)
        self.host_norm = nn.LayerNorm(host_feature_dim)
        self.host = nn.Linear(host_feature_dim, hidden_dim)
        self.protocol = nn.Embedding(protocol_vocab_size, hidden_dim)
        self.service = nn.Embedding(service_vocab_size, hidden_dim)
        self.port = nn.Embedding(port_buckets, hidden_dim)
        self.port_buckets = port_buckets

    def forward(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        return {
            "flow": self.flow(self.flow_norm(graph["flow"].x)),
            "host": self.host(self.host_norm(graph["host"].x)),
            "protocol": self.protocol(graph["protocol"].vocab_id),
            "service": self.service(graph["service"].vocab_id),
            "port": self.port(port_bucket(graph["port"].port_number, self.port_buckets)),
        }


@dataclass
class RelationSpecificOutput:
    """One layer's (or the whole model's) output.

    - ``relations``: ``{node_type: {relation_name: Tensor[N, hidden_dim]}}``
      -- every relation-specific embedding, kept separate.
    - ``fused``: ``{node_type: Tensor[N, hidden_dim]}`` -- the attention-
      fused embedding per node type, used as the next layer's input (or the
      final representation, for the last layer).
    - ``attention``: ``{node_type: {relation_name: beta}}`` -- the softmax
      relation-importance weight each relation received during fusion.
    """

    relations: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    fused: dict[str, torch.Tensor] = field(default_factory=dict)
    attention: dict[str, dict[str, float]] = field(default_factory=dict)


class RelationSpecificLayer(nn.Module):
    """One round of relation-specific message passing + semantic attention fusion."""

    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        node_types: list[str],
        hidden_dim: int,
        attn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.conv = RelationSpecificConv(edge_types, hidden_dim)
        self.fusion = nn.ModuleDict(
            {node_type: SemanticAttention(hidden_dim, attn_dim) for node_type in node_types}
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> RelationSpecificOutput:
        relation_embeds = self.conv(x_dict, edge_index_dict)
        fused: dict[str, torch.Tensor] = {}
        attention: dict[str, dict[str, float]] = {}
        for node_type, per_relation in relation_embeds.items():
            if not per_relation:
                # No incoming relation this layer (shouldn't happen on the
                # current schema) -- pass the node's embedding through
                # unchanged rather than losing it.
                fused[node_type] = x_dict[node_type]
                attention[node_type] = {}
                continue
            names = list(per_relation.keys())
            embeds = [per_relation[name] for name in names]
            fused_x, beta = self.fusion[node_type](embeds)
            fused[node_type] = fused_x
            attention[node_type] = dict(zip(names, beta.tolist(), strict=True))
        return RelationSpecificOutput(relations=relation_embeds, fused=fused, attention=attention)


class RelationSpecificHeteroGNN(nn.Module):
    """Full Step 3 encoder: raw-feature encoding -> N relation-specific +
    attention-fusion layers.

    Operates directly on a graph as loaded from disk -- no edge transform
    needed before calling ``forward``. Build with ``from_graph`` rather than
    the constructor directly in normal use -- it reads the edge/node types
    straight from a sample graph so the relation-specific weight set always
    matches the schema being trained on.
    """

    def __init__(
        self,
        edge_types: list[tuple[str, str, str]],
        node_types: list[str],
        hidden_dim: int,
        protocol_vocab_size: int,
        service_vocab_size: int,
        num_layers: int = 1,
        attn_dim: int = 128,
        port_buckets: int = 32,
    ) -> None:
        super().__init__()
        self.encoders = NodeFeatureEncoders(
            hidden_dim, protocol_vocab_size, service_vocab_size, port_buckets=port_buckets
        )
        self.layers = nn.ModuleList(
            [
                RelationSpecificLayer(edge_types, node_types, hidden_dim, attn_dim)
                for _ in range(num_layers)
            ]
        )

    @classmethod
    def from_graph(
        cls,
        sample_graph: HeteroData,
        hidden_dim: int,
        protocol_vocab_size: int,
        service_vocab_size: int,
        num_layers: int = 1,
        attn_dim: int = 128,
        port_buckets: int = 32,
    ) -> RelationSpecificHeteroGNN:
        """Build a model whose relation-specific weights match `sample_graph`'s
        metadata exactly -- metadata is fixed at construction time since
        every mini-graph in this benchmark shares the same 5 node types / 6
        one-directional relations."""
        node_types, edge_types = sample_graph.metadata()
        return cls(
            edge_types,
            node_types,
            hidden_dim,
            protocol_vocab_size,
            service_vocab_size,
            num_layers,
            attn_dim,
            port_buckets,
        )

    def forward(self, graph: HeteroData) -> RelationSpecificOutput:
        x_dict = self.encoders(graph)
        edge_index_dict = graph.edge_index_dict
        output = RelationSpecificOutput(fused=x_dict)
        for layer in self.layers:
            output = layer(output.fused, edge_index_dict)
        return output
