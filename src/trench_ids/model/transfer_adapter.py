"""Transfer Adapter for Task-Conditioned Transfer Representation Learning (TCTRL).

The adapter learns to modify relation-specific representations based on
the incoming task signature, enabling faster adaptation to new tasks.

Architecture:
    z'_r = z_r + g_r(q) * A_r(z_r)
    
where:
    z_r = relation-specific embedding from frozen backbone
    q = label-free task signature
    A_r = relation-specific adapter (zero-initialized output)
    g_r = task-conditioned gate (scalar per relation)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from trench_ids.constants import FLOW_RELATIONS


class TransferAdapter(nn.Module):
    """Task-Conditioned Transfer Representation Adapter.
    
    Learns to adapt relation-specific representations based on
    the incoming task signature for faster future-task adaptation.
    
    Forward pass:
        z'_r = z_r + g_r(q) * A_r(z_r)
        
    where:
        z_r = relation-specific embedding from frozen backbone
        q = label-free task signature (137 dims in B0)
        A_r = relation-specific adapter (zero-init output layer)
        g_r = task-conditioned gate (scalar per relation, sigmoid)
    
    Identity initialization: A_r's final layer is zero-initialized
    so that at initialization z'_r ≈ z_r (adapter behaves like baseline).
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        adapter_dim: int = 32,
        num_relations: int = len(FLOW_RELATIONS),
        signature_dim: int = 137,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.num_relations = num_relations
        self.signature_dim = signature_dim
        
        # Per-relation adapter: hidden_dim -> adapter_dim -> hidden_dim
        # Final layer ZERO-INITIALIZED so A_r(z) ≈ 0 at init
        self.relation_adapters = nn.ModuleDict({
            r: nn.Sequential(
                nn.Linear(hidden_dim, adapter_dim),
                nn.ReLU(inplace=True),
                nn.Linear(adapter_dim, hidden_dim)
            ) for r in FLOW_RELATIONS
        })
        
        # Task-conditioned gate: signature -> [0,1] per relation
        self.task_encoder = nn.Sequential(
            nn.Linear(137, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(FLOW_RELATIONS))  # one scalar per relation
        )
        
        # Initialize: zero-init final adapter layer
        self._init_identity()
    
    def _init_identity(self):
        """Zero-initialize final adapter layer for identity initialization.
        
        This ensures A_r(z_r) ≈ 0 at initialization, so the adapter
        starts behaving exactly like the baseline (no adapter).
        """
        for r in FLOW_RELATIONS:
            adapter = self.relation_adapters[r]
            # Final linear layer is the last module
            final_layer = adapter[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)
    
    def forward(
        self,
        relation_embeds: Dict[str, torch.Tensor],
        task_signature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with current parameters.
        
        Args:
            relation_embeds: Dict mapping relation name to embeddings [N, hidden_dim]
            task_signature: Task signature tensor [137] or [B, 137]
        
        Returns:
            Dict mapping relation name to adapted embeddings [N, hidden_dim]
        """
        if not relation_embeds:
            return {}

        embed_device = next(iter(relation_embeds.values())).device
        task_signature = task_signature.to(embed_device)

        # Compute gates from task signature
        # task_signature: [137] or [B, 137] -> gates: [num_relations]
        if task_signature.dim() == 1:
            task_signature = task_signature.unsqueeze(0)  # [1, 137]
        
        gates = torch.sigmoid(self.task_encoder(task_signature))  # [1, num_relations] or [B, num_relations]
        gates = gates.squeeze(0) if gates.size(0) == 1 else gates  # [num_relations] or [B, num_relations]
        
        adapted = {}
        for i, (r, emb) in enumerate(relation_embeds.items()):
            if r not in FLOW_RELATIONS:
                adapted[r] = emb
                continue
            
            delta = self.relation_adapters[r](emb)  # [N, hidden_dim]
            gate = gates[i]  # scalar or [B]
            
            # Reshape gate for broadcasting
            if gate.dim() == 0:
                gate = gate.view(1, 1)  # [1, 1]
            elif gate.dim() == 1:
                gate = gate.unsqueeze(-1)  # [B, 1]
            
            # Residual with task-conditioned scaling
            adapted[r] = emb + gate * delta
        
        return adapted
    
    def forward_with_params(
        self,
        relation_embeds: Dict[str, torch.Tensor],
        task_signature: torch.Tensor,
        params: dict,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using explicit parameter dict (for FOMAML inner loop).
        
        Args:
            relation_embeds: Dict mapping relation name to embeddings [N, hidden_dim]
            task_signature: Task signature tensor [137] or [B, 137]
            params: Dict mapping parameter names to tensors
        
        Returns:
            Dict mapping relation name to adapted embeddings [N, hidden_dim]
        """
        if not relation_embeds:
            return {}

        embed_device = next(iter(relation_embeds.values())).device
        task_signature = task_signature.to(embed_device)

        # Extract task encoder params
        task_encoder_params = {k: v for k, v in params.items() if 'task_encoder' in k}
        relation_adapter_params = {k: v for k, v in params.items() if 'relation_adapters' in k}
        
        # Compute gates
        if task_signature.dim() == 1:
            task_signature = task_signature.unsqueeze(0)
        
        gates = torch.sigmoid(
            F.linear(
                F.relu(F.linear(task_signature, task_encoder_params['task_encoder.0.weight'], 
                               task_encoder_params.get('task_encoder.0.bias'))),
                task_encoder_params['task_encoder.2.weight'],
                task_encoder_params.get('task_encoder.2.bias')
            )
        )
        gates = gates.squeeze(0) if gates.size(0) == 1 else gates
        
        adapted = {}
        for i, (r, emb) in enumerate(relation_embeds.items()):
            if r not in FLOW_RELATIONS:
                adapted[r] = emb
                continue
            
            # Extract adapter params for this relation
            prefix = f'relation_adapters.{r}.'
            adapter_params = {k[len(prefix):]: v for k, v in relation_adapter_params.items() if k.startswith(prefix)}
            
            # Forward through adapter
            x = F.linear(emb, adapter_params['0.weight'], adapter_params.get('0.bias'))
            x = F.relu(x)
            delta = F.linear(x, adapter_params['2.weight'], adapter_params.get('2.bias'))
            
            gate = gates[i]
            if gate.dim() == 0:
                gate = gate.view(1, 1)
            elif gate.dim() == 1:
                gate = gate.unsqueeze(-1)
            
            adapted[r] = emb + gate * delta
        
        return adapted
    
    def get_param_dict(self) -> dict:
        """Get parameter dict for FOMAML inner loop."""
        return {name: param for name, param in self.named_parameters()}


class RandomTransferAdapter(TransferAdapter):
    """Randomly initialized adapter (capacity control).
    
    Same architecture as TransferAdapter but not meta-trained.
    Used as capacity control in B0.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Random initialization already done by parent
        # Do NOT call _init_identity - we want random weights
        for r in FLOW_RELATIONS:
            adapter = self.relation_adapters[r]
            final_layer = adapter[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.xavier_uniform_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)


def fuse_adapted(adapted_embeds: dict, fusion_module=None, residual: torch.Tensor = None, node_type: str = 'flow') -> torch.Tensor:
    """Fuse adapted relation embeddings using the model's fusion module.
    
    Args:
        adapted_embeds: Dict mapping relation name to embeddings [N, hidden_dim]
        fusion_module: The model's fusion module (e.g., model.layers[-1].fusion[node_type])
        residual: Residual connection to add after fusion (layer N-1 fused output).
                  If None, no residual is added.
        node_type: Node type to fuse (default: 'flow')
    
    Returns:
        Fused embeddings [N, hidden_dim] - same dimension as backbone's fused output
    """
    if fusion_module is None:
        # Fallback: simple mean if no fusion module provided
        embeddings = [adapted_embeds[r] for r in FLOW_RELATIONS if r in adapted_embeds]
        return torch.stack(embeddings, dim=0).mean(0)
    
    # Use the model's SemanticAttention fusion module
    # IMPORTANT: Use FLOW_RELATIONS order (matches model's forward pass), NOT sorted()
    embeddings = [adapted_embeds[r] for r in FLOW_RELATIONS if r in adapted_embeds]
    fused, _ = fusion_module(embeddings)
    
    # Add residual connection if provided (for multi-layer models with use_residual=True)
    if residual is not None:
        fused = fused + residual
    
    return fused


class LightweightClassifierHead(nn.Module):
    """Lightweight classifier head for target-task adaptation.
    
    Used identically across all B0 conditions for fair comparison.
    """
    
    def __init__(self, input_dim: int, num_classes: int = 11):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def create_classifier_head(input_dim: int, num_classes: int = 11) -> LightweightClassifierHead:
    """Factory for creating identical classifier heads across conditions."""
    return LightweightClassifierHead(input_dim, num_classes)


if __name__ == "__main__":
    # Quick test
    adapter = TransferAdapter()
    print("TransferAdapter created successfully")
    print(f"Parameters: {sum(p.numel() for p in adapter.parameters())}")
    
    # Test zero-init
    for r in FLOW_RELATIONS:
        final_layer = adapter.relation_adapters[r][-1]
        if isinstance(final_layer, nn.Linear):
            assert final_layer.weight.abs().max() < 1e-6, "Not zero-initialized!"
            assert final_layer.bias.abs().max() < 1e-6, "Not zero-initialized!"
    print("Zero-init verified")
    
    # Test forward
    embeds = {r: torch.randn(10, 64) for r in FLOW_RELATIONS}
    sig = torch.randn(137)
    out = adapter(embeds, sig)
    print(f"Output shapes: {[(k, v.shape) for k, v in out.items()]}")
    
    # Test random adapter
    rand_adapter = RandomTransferAdapter()
    for r in FLOW_RELATIONS:
        final_layer = rand_adapter.relation_adapters[r][-1]
        if isinstance(final_layer, nn.Linear):
            assert final_layer.weight.abs().max() > 1e-6, "Not random!"
    print("Random adapter verified")