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


class ConcatFusion(nn.Module):
    """Fuse relation embeddings by concatenation instead of a weighted sum.

    Why this exists. ``SemanticAttention`` above produces a *convex*
    combination: ``beta`` is a softmax, so the fused vector is a weighted
    average of the R relation embeddings and lands inside their convex hull.
    Two consequences, both measured rather than argued:

    1. Information from distinct relations is averaged together and cannot be
       recovered downstream. R x hidden_dim values are compressed to
       hidden_dim before anything reads them.
    2. ``beta`` is one scalar per relation, averaged over all nodes, so every
       node in the batch is fused with the identical mixture. The fusion
       cannot say "this relation matters for this flow".

    On the 2026-08-17 benchmark this was the architecture ladder's whole
    story. Flat+Host -- per-flow features with both endpoint hosts'
    aggregates *concatenated* and read by an MLP, no message passing at all --
    scored 0.854 mean final accuracy against the GNN's 0.749, at matched
    parameter count and on the same replay setting. The flat baseline was
    already documented as concatenating "so the baseline is not handicapped
    relative to the GNN by an information-losing merge"; the ladder result
    says the GNN is the arm taking that hit, and that propagation gains less
    than the convex combination destroys.

    So this variant changes exactly one thing: concatenate the R embeddings
    ([N, R*hidden_dim]) and let a learned projection do the mixing. That makes
    the GNN's fusion at least as expressive as the flat model's, while keeping
    message passing. It is the minimal edit that distinguishes "message
    passing does not help on this data" from "message passing helps but the
    merge threw it away" -- two very different findings that the ladder alone
    cannot separate.

    ``beta`` is returned uniform (1/R). Concatenation has no scalar
    relation-importance weight by construction: the mixing lives inside
    ``project``'s weight matrix. That is a deliberate loss of one
    interpretability handle, and it costs nothing the project depends on --
    nothing in ``src/`` reads ``RelationSpecificOutput.attention``, and the
    relation-importance signal the CL steps actually consume comes from
    ``transferability.py``'s per-relation cosine scores and ``drift.py``'s
    per-relation drift, both computed from ``relations``, which this leaves
    untouched.

    ``num_relations == 1`` is the identity case (``forward`` returns the
    single embedding unchanged, matching a HAN-style fusion's single-relation
    ``beta=1.0`` shortcut), so ``project`` is not built at all rather than
    built and left unreachable -- this schema gives Protocol/Service/Port
    exactly one incoming relation each, so an allocated-but-unused
    ``Linear(hidden_dim, hidden_dim)`` per node type (4,160 params each at
    ``hidden_dim=64``, 12,480 total) previously inflated
    ``model.parameters()`` without the classification loss (or any loss) ever
    reaching it -- the same "counted but gradient-unreachable" defect already
    diagnosed for the attention-fusion baseline's six dead non-Flow
    relations, reproduced here by a different mechanism.
    """

    def __init__(self, hidden_dim: int, num_relations: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.project = (
            nn.Linear(num_relations * hidden_dim, hidden_dim) if num_relations > 1 else None
        )

    def forward(self, embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not embeddings:
            raise ValueError("ConcatFusion requires at least one relation embedding")
        if len(embeddings) == 1:
            return embeddings[0], torch.ones(1, device=embeddings[0].device)
        if len(embeddings) != self.num_relations:
            raise ValueError(
                f"ConcatFusion built for {self.num_relations} relations, got {len(embeddings)}. "
                "The projection is sized at construction, so a node type's relation count "
                "cannot change between build and forward."
            )
        fused = self.project(torch.cat(embeddings, dim=-1))
        uniform = torch.full(
            (len(embeddings),), 1.0 / len(embeddings), device=embeddings[0].device
        )
        return fused, uniform
