"""HAN-style semantic-level attention fusion across relation-specific embeddings.

Wang et al., "Heterogeneous Graph Attention Network" (WWW 2019) computes one
importance weight per *relation* (not per node): each relation's embeddings
are projected through a shared MLP + learnable query vector, averaged across
all nodes of that type, then softmax-normalized across relations. That
(rather than a node-conditioned/GAT-style per-node-per-relation weight) is
the right granularity here, because a single scalar-per-relation weight is
exactly the "relation importance" signal the project's later continual-
learning steps need (transferability estimation + a relation-aware EWC, see
CLAUDE.md's "Proposed pipeline" steps 5-7) -- it is directly reusable rather
than something that would need to be summarized from a node-level attention
map later. Trade-off: it cannot express "this relation matters more for node
A than node B"; if that turns out to matter, extend to a node-conditioned
variant then (concatenate each node's own embedding into the attention
query) rather than building it now on spec.
"""

from __future__ import annotations

import torch
from torch import nn


class SemanticAttention(nn.Module):
    """Fuse a variable number of same-shaped relation embeddings into one.

    ``forward`` returns ``(fused, beta)``: ``fused`` is
    ``[num_nodes, hidden_dim]``; ``beta`` is a ``[num_relations]`` tensor of
    softmax-normalized relation weights, in the same order as the input list
    -- kept for later inspection (e.g. logging which relation a node type
    currently leans on, or feeding a future relation-importance estimator).
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.Tanh())
        self.query = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not embeddings:
            raise ValueError("SemanticAttention requires at least one relation embedding")
        if len(embeddings) == 1:
            return embeddings[0], torch.ones(1, device=embeddings[0].device)

        stacked = torch.stack(embeddings, dim=0)  # [R, N, hidden_dim]
        scores = self.query(self.project(stacked)).squeeze(-1)  # [R, N]
        relation_scores = scores.mean(dim=1)  # [R] -- HAN-style: average over nodes first
        beta = torch.softmax(relation_scores, dim=0)  # [R]
        fused = (beta.view(-1, 1, 1) * stacked).sum(dim=0)  # [N, hidden_dim]
        return fused, beta
