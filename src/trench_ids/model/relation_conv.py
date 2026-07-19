"""Relation-specific message passing for the TRENCH-IDS heterogeneous graph.

Each edge type gets its own learnable linear transform (a relation-specific
weight matrix), applied to the source node's embedding before aggregating at
the destination. Unlike ``torch_geometric.nn.HeteroConv`` (which sums/means
every incoming relation into one embedding per destination node type before
returning), this module deliberately keeps each relation's contribution
separate -- see ``trench_ids.model.attention_fusion.SemanticAttention`` for
how they are later combined. Keeping them separate is what lets a later
continual-learning component (the relation-specific memory bank /
transferability estimation from CLAUDE.md's proposed pipeline) inspect a
single relation's embedding on its own instead of an already-blended one.

**Per-relation self-concatenation** (professor's instruction, 2026-07-19):
before this, a relation's stored embedding was 100% neighbor-aggregated
signal -- the destination node's own features never entered it, worst for
Flow (the classification target). Now, after aggregating each relation's
neighbor messages, the destination node's own current embedding
(``x_dict[dst_type]``) is concatenated onto it and passed through a second
per-relation linear layer (``combine_lins``) back down to ``hidden_dim``.
This is GraphSAGE's concatenation aggregator (Hamilton et al., NeurIPS 2017,
"Inductive Representation Learning on Large Graphs"):
``h_v = W * CONCAT(h_v, AGGREGATE({h_u : u in N(v)}))``, applied once per
relation rather than once per node overall -- so every relation-specific
embedding already carries the node's own signal, and
``SemanticAttention`` fusion needs no separate self-term (it has none).
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter


def edge_type_key(edge_type: tuple[str, str, str]) -> str:
    """Stable string key for an edge type triple (``nn.ModuleDict`` keys must be str)."""
    return "__".join(edge_type)


class RelationSpecificConv(nn.Module):
    """One relation-specific linear transform + mean-aggregation + self-concat per edge type.

    ``forward`` returns, for every destination node type, a dict of
    ``{relation_name: Tensor[num_nodes_of_that_type, hidden_dim]}`` -- one
    tensor per *incoming* relation, not summed together. Relation names are
    each edge type's middle element (e.g. ``"originates"``,
    ``"terminates_at"``); this is unique per destination node type for the
    current TRENCH-IDS schema (5 node types, 11 relations -- the original 6
    one-directional relations plus 5 reverse relations added so Flow has 5
    incoming relations instead of 1 and Host has 3 instead of 2, see
    ``trench_ids.graphs``' module docstring) -- if a future schema change
    introduces two relations with the same name into the same destination
    type, key by the full edge-type tuple instead.

    Each relation's embedding is ``combine_lins[relation](CONCAT(dst's own
    embedding, mean-aggregated transformed source messages))`` -- see the
    module docstring for why (GraphSAGE-style per-relation self-preservation).
    """

    def __init__(self, edge_types: list[tuple[str, str, str]], hidden_dim: int) -> None:
        super().__init__()
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.rel_lins = nn.ModuleDict(
            {edge_type_key(et): nn.Linear(hidden_dim, hidden_dim) for et in self.edge_types}
        )
        self.combine_lins = nn.ModuleDict(
            {edge_type_key(et): nn.Linear(2 * hidden_dim, hidden_dim) for et in self.edge_types}
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        relation_embeds: dict[str, dict[str, torch.Tensor]] = {
            node_type: {} for node_type in x_dict
        }
        for edge_type in self.edge_types:
            src_type, relation, dst_type = edge_type
            edge_index = edge_index_dict.get(edge_type)
            if edge_index is None or edge_index.numel() == 0:
                continue
            lin = self.rel_lins[edge_type_key(edge_type)]
            transformed_src = lin(x_dict[src_type])
            messages = transformed_src[edge_index[0]]
            num_dst = x_dict[dst_type].size(0)
            aggregated = scatter(messages, edge_index[1], dim=0, dim_size=num_dst, reduce="mean")
            combined = torch.cat([x_dict[dst_type], aggregated], dim=-1)
            relation_embeds[dst_type][relation] = self.combine_lins[edge_type_key(edge_type)](
                combined
            )
        return relation_embeds
