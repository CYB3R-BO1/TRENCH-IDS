"""Stability Regularization: constrain relation-specific parameter changes
during continual learning to test if parameter stability causally improves
forward transfer.

This implements the Stage B experiment: add L2 penalty on parameter deviation
from the pre-task checkpoint for specified relations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from trench_ids.constants import FLOW_RELATIONS
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN


@dataclass
class StabilityConfig:
    """Configuration for stability regularization."""
    enabled: bool = False
    lambda_stability: float = 1.0
    # Which relations to constrain: "all", "uniform", "targeted_by", "oracle"
    mode: str = "uniform"
    # For oracle mode: relation -> weight (e.g., from T_r)
    oracle_weights: dict[str, float] | None = None
    # Normalize by parameter norm (recommended)
    normalize: bool = True
    epsilon: float = 1e-8


def extract_relation_parameters(
    model: RelationSpecificHeteroGNN,
    relation: str,
) -> list[torch.nn.Parameter]:
    """Extract all parameters for a specific relation (rel_lins + combine_lins)."""
    params = []
    for layer in model.layers:
        conv = layer.conv
        # rel_lins
        for edge_key, module in conv.rel_lins.items():
            if f"__{relation}__flow" in edge_key:
                params.extend(list(module.parameters()))
        # combine_lins
        for edge_key, module in conv.combine_lins.items():
            if f"__{relation}__flow" in edge_key:
                params.extend(list(module.parameters()))
    return params


def get_all_relation_parameters(
    model: RelationSpecificHeteroGNN,
    relations: list[str] = None,
) -> dict[str, list[torch.nn.Parameter]]:
    """Get parameter dict for all specified relations."""
    if relations is None:
        relations = FLOW_RELATIONS
    return {r: extract_relation_parameters(model, r) for r in relations}


class StabilityRegularizer(nn.Module):
    """Stability regularization module.
    
    Computes L2 penalty: ||theta - theta_old||^2 / (||theta_old||^2 + eps)
    for each relation's parameters.
    """
    
    def __init__(
        self,
        model: RelationSpecificHeteroGNN,
        config: StabilityConfig,
        device: torch.device,
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # Store reference parameters (frozen)
        self.relations = FLOW_RELATIONS if config.mode in ("uniform", "all") else [config.mode]
        if config.mode == "oracle" and config.oracle_weights:
            # Include all relations that have non-zero oracle weights
            self.relations = [r for r, w in config.oracle_weights.items() if w > 0]
        
        self.ref_params = {}
        for relation in self.relations:
            params = extract_relation_parameters(model, relation)
            if params:
                self.ref_params[relation] = [p.detach().clone() for p in params]
    
    def forward(self, model: RelationSpecificHeteroGNN) -> torch.Tensor:
        """Compute total stability penalty."""
        if not self.config.enabled or not self.ref_params:
            return torch.tensor(0.0, device=self.device)
        
        total_penalty = torch.tensor(0.0, device=self.device)
        
        for relation, ref_param_list in self.ref_params.items():
            current_params = extract_relation_parameters(model, relation)
            if not current_params:
                continue
            
            for curr, ref in zip(current_params, ref_param_list):
                diff = curr - ref
                if self.config.normalize:
                    ref_norm = ref.norm() + self.config.epsilon
                    penalty = (diff.norm() / ref_norm) ** 2
                else:
                    penalty = diff.norm() ** 2
                total_penalty = total_penalty + penalty
        
        return self.config.lambda_stability * total_penalty


def create_stability_regularizer(
    model: RelationSpecificHeteroGNN,
    mode: str,
    lambda_stability: float,
    oracle_weights: dict[str, float] = None,
    device: torch.device = None,
) -> StabilityRegularizer:
    """Factory function for stability regularizer."""
    if device is None:
        device = next(model.parameters()).device
    
    config = StabilityConfig(
        enabled=(mode != "none" and lambda_stability > 0),
        lambda_stability=lambda_stability,
        mode=mode,
        oracle_weights=oracle_weights,
    )
    
    return StabilityRegularizer(model, config, device)


def compute_parameter_change(
    model: RelationSpecificHeteroGNN,
    ref_params: dict[str, list[torch.Tensor]],
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Compute parameter change for each relation relative to reference."""
    changes = {}
    for relation, ref_param_list in ref_params.items():
        current_params = extract_relation_parameters(model, relation)
        if not current_params:
            changes[relation] = 0.0
            continue
        
        total_diff_norm = 0.0
        total_ref_norm = 0.0
        
        for curr, ref in zip(current_params, ref_param_list):
            diff = (curr - ref).detach()
            total_diff_norm += diff.norm().item() ** 2
            total_ref_norm += ref.norm().item() ** 2
        
        if normalize:
            changes[relation] = (total_diff_norm ** 0.5) / (total_ref_norm ** 0.5 + epsilon)
        else:
            changes[relation] = total_diff_norm ** 0.5
    
    return changes


if __name__ == "__main__":
    # Quick test
    from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
    from trench_ids.vocab import load_vocab
    import torch_geometric.data
    
    # This is just a syntax check
    print("Stability module loaded successfully")