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
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter


def edge_type_key(edge_type: tuple[str, str, str]) -> str:
    """Stable string key for an edge type triple (``nn.ModuleDict`` keys must be str)."""
    return "__".join(edge_type)


class RelationSpecificConv(nn.Module):
    """One relation-specific linear transform + mean-aggregation per edge type.

    ``forward`` returns, for every destination node type, a dict of
    ``{relation_name: Tensor[num_nodes_of_that_type, hidden_dim]}`` -- one
    tensor per *incoming* relation, not summed together. Relation names are
    each edge type's middle element (e.g. ``"originates"``,
    ``"rev_terminates_at"``); this is unique per destination node type for
    the current TRENCH-IDS schema (5 node types, 6 relations, doubled to 11
    directed edge types by ``trench_ids.model.rhgnn.to_bidirectional``) -- if
    a future schema change introduces two relations with the same name into
    the same destination type, key by the full edge-type tuple instead.
    """

    def __init__(self, edge_types: list[tuple[str, str, str]], hidden_dim: int) -> None:
        super().__init__()
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.rel_lins = nn.ModuleDict(
            {edge_type_key(et): nn.Linear(hidden_dim, hidden_dim) for et in self.edge_types}
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
            relation_embeds[dst_type][relation] = aggregated
        return relation_embeds
