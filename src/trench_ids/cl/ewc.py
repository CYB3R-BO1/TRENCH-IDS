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
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader

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


class OnlineEWCState:
    """Running Fisher + reference-parameter (theta*) snapshot for one
    parameter group, Online EWC style (Schwarz et al. 2018) -- constant
    memory regardless of how many tasks have been seen, since each
    ``update()`` blends the new Fisher estimate into the running one rather
    than storing a separate Fisher per task."""

    def __init__(self, parameter_names: list[str], gamma: float) -> None:
        self.parameter_names = list(parameter_names)
        self.gamma = gamma
        self.fisher: dict[str, torch.Tensor] = {}
        self.theta_star: dict[str, torch.Tensor] = {}
        self.initialized = False

    def update(
        self, fisher_estimate: dict[str, torch.Tensor], current_params: dict[str, torch.Tensor]
    ) -> None:
        for name in self.parameter_names:
            new_fisher = fisher_estimate[name].detach()
            if name in self.fisher:
                self.fisher[name] = self.gamma * self.fisher[name] + new_fisher
            else:
                self.fisher[name] = new_fisher.clone()
            self.theta_star[name] = current_params[name].detach().clone()
        self.initialized = True

    def penalty(self, current_params: dict[str, torch.Tensor]) -> torch.Tensor:
        any_param = current_params[self.parameter_names[0]]
        if not self.initialized:
            return torch.zeros((), device=any_param.device, dtype=any_param.dtype)
        total = torch.zeros((), device=any_param.device, dtype=any_param.dtype)
        for name in self.parameter_names:
            diff = current_params[name] - self.theta_star[name]
            total = total + (self.fisher[name] * diff.pow(2)).sum()
        return total


def estimate_fisher(
    model: nn.Module,
    classifier: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """One evaluation-mode forward/backward pass (no optimizer step) over
    ``dataloader``, covering *every* named parameter across ``model`` and
    ``classifier`` at once -- deliberately not called per parameter group,
    so a task transition costs exactly one pass over the data regardless of
    how the 12 EWC groups later partition the result (design §2). Squared
    per-batch gradients are accumulated and averaged over the number of
    batches -- the batch-gradient-squared Fisher approximation, not the true
    per-example empirical Fisher."""
    model_was_training = model.training
    classifier_was_training = classifier.training
    model.eval()
    classifier.eval()

    params = all_named_parameters(model, classifier)
    squared_grad_sums = {name: torch.zeros_like(p) for name, p in params.items()}
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        for p in params.values():
            p.grad = None
        output = model(batch)
        logits = classifier(output.fused["flow"])
        loss = F.cross_entropy(logits, batch["flow"].y)
        loss.backward()
        for name, p in params.items():
            if p.grad is not None:
                squared_grad_sums[name] += p.grad.detach().pow(2)
        num_batches += 1

    if model_was_training:
        model.train()
    if classifier_was_training:
        classifier.train()

    if num_batches == 0:
        return squared_grad_sums
    return {name: total / num_batches for name, total in squared_grad_sums.items()}


class OnlineEWCManager:
    """Owns all 12 ``OnlineEWCState`` instances (1 shared + 5 Flow relations
    + 6 other relations) so ``train.py`` only calls ``loss()``/``update_all()``
    instead of coordinating 12 objects itself (design §2's approved code
    structure)."""

    def __init__(
        self,
        model: nn.Module,
        classifier: nn.Module,
        flow_relations: list[str],
        gamma: float,
        lambda_r: float,
        lambda_s: float,
        lambda_u: float,
    ) -> None:
        self.flow_relations = list(flow_relations)
        self.lambda_r = lambda_r
        self.lambda_s = lambda_s
        self.lambda_u = lambda_u
        self.groups = partition_parameter_names(model, classifier, flow_relations)
        self.states = {
            group_name: OnlineEWCState(names, gamma) for group_name, names in self.groups.items()
        }

    def loss(
        self, model: nn.Module, classifier: nn.Module, w_r: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        params = all_named_parameters(model, classifier)
        total = self.lambda_s * self.states["shared"].penalty(params)
        for group_name, state in self.states.items():
            if group_name == "shared":
                continue
            if group_name in self.flow_relations:
                total = total + self.lambda_r * w_r[group_name] * state.penalty(params)
            else:
                total = total + self.lambda_u * state.penalty(params)
        return total

    def update_all(
        self, model: nn.Module, classifier: nn.Module, dataloader: DataLoader, device: torch.device
    ) -> None:
        fisher_estimate = estimate_fisher(model, classifier, dataloader, device)
        params = all_named_parameters(model, classifier)
        for state in self.states.values():
            state.update(fisher_estimate, params)
