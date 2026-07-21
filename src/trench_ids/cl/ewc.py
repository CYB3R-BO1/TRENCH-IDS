"""Steps 6-8 — relation-aware Online EWC (Schwarz et al. 2018).

Three parameter-group categories, per
``docs/superpowers/specs/2026-07-21-relation-aware-ewc-design.md`` §1:
"shared" modules (NodeFeatureEncoders, SemanticAttention fusion, the
classifier head), Flow's 5 incoming relations (transferability-weighted,
the method's novel contribution), and the remaining 6 relation-specific
groups (standard, unweighted EWC). Grouping is derived purely from each
parameter's name so it works for any ``num_layers`` -- a relation's
parameters across every layer land in the same group.
"""

from __future__ import annotations

import re

import torch
from torch import nn

FLOW_RELATIONS = ["originates", "terminated_by", "targeted_by", "protocol_of", "service_of"]

_REL_PARAM_RE = re.compile(r"^layers\.\d+\.conv\.(?:rel_lins|combine_lins)\.([^.]+)\.")


def all_named_parameters(model: nn.Module, classifier: nn.Module) -> dict[str, torch.Tensor]:
    """Every trainable parameter across both modules, keyed by a
    ``"model."``/``"classifier."``-prefixed name -- the shared addressing
    scheme ``OnlineEWCState``/``OnlineEWCManager`` use throughout, since the
    encoder and classifier are two separate ``nn.Module``s in ``train.py``."""
    params = {f"model.{name}": p for name, p in model.named_parameters()}
    params.update({f"classifier.{name}": p for name, p in classifier.named_parameters()})
    return params


def partition_parameter_names(
    model: nn.Module, classifier: nn.Module, flow_relations: list[str]
) -> dict[str, list[str]]:
    """Split every parameter name into one of the three EWC group categories.

    Returns ``{"shared": [...], <flow_relation_name>: [...], ...,
    "other::<relation_name>": [...], ...}``. A relation-specific parameter
    (``rel_lins``/``combine_lins`` under any layer) is matched by name via
    ``_REL_PARAM_RE``; everything else (encoders, fusion, classifier) is
    "shared". The edge-type key embedded in a relation parameter's name is
    always ``"src__relation__dst"`` (``trench_ids.model.relation_conv.edge_type_key``),
    so splitting on ``"__"`` recovers the relation name directly.
    """
    groups: dict[str, list[str]] = {"shared": []}
    for name, _ in model.named_parameters():
        match = _REL_PARAM_RE.match(name)
        if match is None:
            groups["shared"].append(f"model.{name}")
            continue
        edge_key = match.group(1)
        _, relation, _ = edge_key.split("__")
        group_name = relation if relation in flow_relations else f"other::{relation}"
        groups.setdefault(group_name, []).append(f"model.{name}")
    for name, _ in classifier.named_parameters():
        groups["shared"].append(f"classifier.{name}")
    return groups
