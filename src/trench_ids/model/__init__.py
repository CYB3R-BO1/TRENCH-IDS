"""Step 3 — relation-specific heterogeneous GNN encoder + attention fusion.

See `trench_ids.model.rhgnn` for the full architecture and design rationale.
"""

from __future__ import annotations

from trench_ids.model.attention_fusion import SemanticAttention
from trench_ids.model.relation_conv import RelationSpecificConv
from trench_ids.model.rhgnn import (
    NodeFeatureEncoders,
    RelationSpecificHeteroGNN,
    RelationSpecificLayer,
    RelationSpecificOutput,
    port_bucket,
)

__all__ = [
    "SemanticAttention",
    "RelationSpecificConv",
    "NodeFeatureEncoders",
    "RelationSpecificHeteroGNN",
    "RelationSpecificLayer",
    "RelationSpecificOutput",
    "port_bucket",
]
