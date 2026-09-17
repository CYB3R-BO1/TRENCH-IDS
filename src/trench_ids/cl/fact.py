"""FACT: Forward Compatible Training for TRENCH-IDS.

Based on: "Forward Compatible Few-Shot Class-Incremental Learning" (Zhou et al., CVPR 2022)

FACT reserves embedding space for future classes by:
1. Virtual Prototypes: Pre-assign learnable virtual class prototypes
2. Virtual Instances: Generate synthetic future-like samples via manifold mixup

The loss L_FACT = L_v + L_f encourages:
- L_v: Compactify known classes, reserve space for virtual prototypes
- L_f: Forecast future classes via manifold mixup, reserve space for them

This is applied during the BASE SESSION (T1) only, per the original paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FACT(nn.Module):
    """FACT module: Virtual Prototypes + Virtual Instances for forward compatibility."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_virtual_classes: int = 10,
        temperature: float = 2.0,
        mixup_alpha: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_virtual_classes = num_virtual_classes
        self.temperature = temperature
        self.mixup_alpha = mixup_alpha

        # Virtual prototypes: reserved embedding space for future classes
        # Shape: [num_virtual_classes, feature_dim]
        self.virtual_prototypes = nn.Parameter(
            torch.randn(num_virtual_classes, feature_dim) * 0.01
        )

    def forward(self, features: torch.Tensor) -> dict:
        """
        Compute FACT losses.

        Args:
            features: Flow embeddings [N, feature_dim] from frozen backbone

        Returns:
            Dict with 'l_virtual' (virtual prototype loss) and 'l_forecast' (forecast loss)
        """
        N = features.shape[0]
        device = features.device

        # Normalize features and virtual prototypes
        features_norm = F.normalize(features, p=2, dim=1)
        v_proto_norm = F.normalize(self.virtual_prototypes, p=2, dim=1)

        # L_v: Virtual Prototype Loss
        # Compute similarities to virtual prototypes
        sim_to_virtual = features_norm @ v_proto_norm.T / self.temperature  # [N, V]

        # For each sample, find nearest virtual prototype
        nearest_virtual = sim_to_virtual.argmax(dim=1)  # [N]

        # L_v: Two parts
        # 1. Standard classification (push sample to its ground-truth class)
        # 2. Virtual loss: push sample toward nearest virtual prototype
        # In base session, we don't have ground-truth "future" classes,
        # so we use the virtual prototype assignment as pseudo-label

        # Virtual prototype loss: encourage samples to be close to their
        # assigned virtual prototype while staying away from others
        # This is similar to the FACT virtual loss L_v

        # For base session (T1), we want to:
        # - Keep real classes compact (handled by standard CE)
        # - Push virtual prototypes away from real class centers
        # - Keep virtual prototypes separated from each other

        # Distance to virtual prototypes
        dist_to_virtual = 1 - sim_to_virtual  # [N, V]

        # L_virtual: encourage separation between virtual prototypes
        # and push real samples away from virtual prototypes
        # (reserving space for future classes)

        # Virtual prototype separation loss
        v_proto_sim = v_proto_norm @ v_proto_norm.T  # [V, V]
        mask = ~torch.eye(self.num_virtual_classes, dtype=torch.bool, device=device)
        l_virtual_separation = v_proto_sim[mask].mean()  # minimize similarity

        # Real samples should NOT be close to virtual prototypes
        # (they should stay in their own class space)
        l_virtual_repulsion = dist_to_virtual.min(dim=1).values.mean()

        # L_v = separation + repulsion
        l_virtual = l_virtual_separation + l_virtual_repulsion

        # L_forecast: Virtual Instances (manifold mixup)
        # Generate synthetic "future-like" samples by mixing real samples
        # and encourage them to be near virtual prototypes

        l_forecast = self._compute_forecast_loss(features_norm, v_proto_norm)

        return {
            'l_virtual': l_virtual,
            'l_forecast': l_forecast,
            'virtual_prototypes': self.virtual_prototypes.detach(),
        }

    def _compute_forecast_loss(self, features_norm: torch.Tensor, v_proto_norm: torch.Tensor) -> torch.Tensor:
        """L_f: Virtual instance loss via manifold mixup."""
        N = features_norm.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=features_norm.device)

        # Manifold mixup: interpolate in feature space
        # Randomly pair samples
        perm = torch.randperm(features_norm.shape[0], device=features_norm.device)
        features_2 = features_norm[perm]

        # Sample mixup coefficient from Beta(alpha, alpha)
        alpha = 1.0  # fixed for now
        lam = torch.distributions.Beta(alpha, alpha).sample([features_norm.shape[0], 1]).to(features_norm.device)

        # Mixed features
        mixed = lam * features_norm + (1 - lam) * features_2

        # Project mixed features toward virtual prototypes
        sim_mixed = mixed @ v_proto_norm.T  # [N, V]

        # Encourage mixed samples to be near virtual prototypes
        # (they represent "future-like" patterns)
        max_sim = sim_mixed.max(dim=1).values
        l_forecast = -max_sim.mean()  # maximize similarity to virtual prototypes

        return l_forecast

    def get_virtual_prototypes(self) -> torch.Tensor:
        """Return current virtual prototypes (detached)."""
        return self.virtual_prototypes.detach()


def fact_loss(model_output, fact_module, task_id: int, is_base_session: bool) -> tuple:
    """
    Compute FACT loss for TRENCH-IDS training.

    Args:
        model_output: Output from RGNN forward pass
        fact_module: FACT module instance
        task_id: Current task ID (1-indexed)
        is_base_session: Whether this is the base session (T1)

    Returns:
        (fact_loss_dict, metrics_dict)
    """
    if not is_base_session:
        return {}, {}

    flow_embeds = model_output.fused["flow"]  # [N, 64]
    losses = fact_module(flow_embeds)

    total_loss = losses['l_virtual'] + losses['l_forecast']

    metrics = {
        'l_virtual': losses['l_virtual'].item(),
        'l_forecast': losses['l_forecast'].item(),
        'total_fact_loss': total_loss.item(),
    }

    return {k: v for k, v in losses.items()}, metrics