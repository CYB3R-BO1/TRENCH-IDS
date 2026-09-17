"""Effective-Rank based Feature Richness (RFR) enhancement for TRENCH-IDS.

Based on: "Improving Forward Compatibility in Class Incremental Learning by 
Increasing Representation Rank and Feature Richness" (Kim et al., Neural Networks, 2024)

RFR increases the effective rank of representations during the base session 
(T1 for TRENCH-IDS), thereby facilitating incorporation of more informative 
features pertinent to unseen novel tasks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def effective_rank(reps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute effective rank (erank) of a batch of representations.
    
    Based on Roy & Vetterli (2007): erank = exp(H) where H is Shannon entropy
    of normalized singular values squared.
    
    Args:
        reps: Representation batch [N, D] 
        eps: Numerical stability constant
        
    Returns:
        Scalar effective rank (continuous, differentiable)
    """
    # L2 normalize each sample
    Z = F.normalize(reps, p=2, dim=1)  # [N, D]
    
    # SVD on normalized representations
    # Note: For large N, consider randomized SVD or using covariance matrix
    _, s, _ = torch.linalg.svd(Z, full_matrices=False)  # s: [min(N,D)]
    
    # Normalized squared singular values = eigenvalues of covariance
    eig_val = (s * s) / reps.shape[0]  # [min(N,D)]
    
    # Shannon entropy of eigenvalue distribution
    # H = -Σ(p_i log p_i) where p_i = eig_val_i
    entropy = - (eig_val * torch.log(eig_val + 1e-8)).sum()
    
    # Effective rank = exp(H)
    erank = torch.exp(entropy)
    
    return erank


def logged_effective_rank(reps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute log(effective_rank) for numerical stability.
    
    This is the quantity used in RFR loss: L_RFR = -log(erank)
    
    Args:
        reps: Representation batch [N, D]
        eps: Numerical stability constant
        
    Returns:
        Scalar log(erank)
    """
    Z = F.normalize(reps, p=2, dim=1)  # [N, D]
    _, s, _ = torch.linalg.svd(Z, full_matrices=False)
    eig_val = (s * s) / reps.shape[0]
    
    # log(erank) = H = -Σ(p_i log p_i)
    logged_erank = - (eig_val * torch.log(eig_val + eps)).sum()
    
    return logged_erank


def rfr_loss(reps: torch.Tensor) -> torch.Tensor:
    """RFR loss: negative log effective rank.
    
    Minimizing this maximizes effective rank, which maximizes 
    Shannon entropy of representations, encouraging feature richness.
    
    Args:
        reps: Representation batch [N, D]
        
    Returns:
        Scalar loss value
    """
    logged_erank = logged_effective_rank(reps)
    return -logged_erank


def relation_aware_rfr_loss(relation_embeds: dict, flow_relations: tuple = None) -> tuple:
    """Relation-aware RFR loss: sum of -log(erank) per relation.
    
    Applied uniformly across all relations (no Stage A weighting).
    
    Args:
        relation_embeds: Dict mapping relation name -> embeddings [N, D]
        flow_relations: Tuple of relation names to consider
        
    Returns:
        (total_loss, per_relation_losses_dict)
    """
    if flow_relations is None:
        flow_relations = (
            'originates', 
            'terminated_by', 
            'targeted_by', 
            'protocol_of', 
            'service_of'
        )
    
    total_loss = 0.0
    per_rel_losses = {}
    count = 0
    
    for r in flow_relations:
        if r in relation_embeds:
            reps = relation_embeds[r]
            # Only compute if we have enough samples
            if reps.shape[0] >= 2:
                rel_loss = rfr_loss(reps)
                total_loss += rel_loss
                per_rel_losses[r] = rel_loss.item()
                count += 1
    
    if count > 0:
        return total_loss / count, per_rel_losses
    else:
        return torch.tensor(0.0, device=next(iter(relation_embeds.values())).device), {}


class RFRMonitor:
    """Monitor effective rank during training for sanity checking."""
    
    def __init__(self):
        self.history = []
        
    def record(self, reps: torch.Tensor, step: int, label: str = ""):
        """Record erank at a training step."""
        with torch.no_grad():
            erank = effective_rank(reps).item()
            logged = logged_effective_rank(reps).item()
        self.history.append({
            'step': step,
            'erank': erank,
            'log_erank': logged,
            'label': label
        })
        return erank
    
    def get_history(self):
        return self.history
    
    def reset(self):
        self.history = []


# Convenience function for TRENCH-IDS integration
def compute_rfr_loss_for_task(model_output, task_id: int, lambda_rfr: float = 0.1, 
                               apply_only_base: bool = True) -> tuple:
    """Compute RFR loss for TRENCH-IDS task.
    
    Args:
        model_output: Output from RGNN forward pass
        task_id: Current task ID (0 = T1 base session)
        lambda_rfr: RFR loss coefficient
        apply_only_base: If True, only apply during task_id == 0 (T1)
        
    Returns:
        (rfr_loss_value, metrics_dict)
    """
    if apply_only_base and task_id != 0:
        return torch.tensor(0.0), {'rfr_loss': 0.0, 'erank': 0.0}
    
    # Get fused flow embeddings from model output
    # model_output.fused['flow'] has shape [N, 64]
    fused_embeds = model_output.fused['flow']
    
    # Compute RFR loss
    loss = rfr_loss(fused_embeds) * lambda_rfr
    
    # Metrics for monitoring
    with torch.no_grad():
        erank_val = effective_rank(fused_embeds).item()
    
    return loss, {'rfr_loss': loss.item(), 'erank': erank_val}