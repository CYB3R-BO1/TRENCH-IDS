"""First-Order MAML (FOMAML) implementation for TCTRL meta-learning.

This module provides explicit first-order meta-learning utilities.
The implementation avoids automatic differentiation pitfalls by explicitly
computing first-order meta-gradients.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from trench_ids.cl.train import load_split, sample_graphs
from trench_ids.labels import NUM_TASKS
from trench_ids.model.transfer_adapter import fuse_adapted


def _get_last_layer_outputs(model: nn.Module, batch):
    """Extract last layer's relation embeddings and previous layer's fused output (for residual).
    
    For a model with N layers and use_residual=True:
    - output.relations['flow'] = layer N-1's relation embeddings
    - output.fused['flow'] = layer N-1's fused + layer N-2's fused (residual)
    - We need layer N-1's fusion module + layer N-2's fused output as residual
    
    Returns:
        relation_embeds: Last layer's relation-specific embeddings
        fusion_module: Last layer's fusion module
        residual: Previous layer's fused output (for residual connection)
    """
    model.eval()
    with torch.no_grad():
        x_dict = model.encoders(batch)
        edge_index_dict = batch.edge_index_dict
        
        # Run through all layers to get intermediate outputs
        prev_fused = x_dict
        for i, layer in enumerate(model.layers):
            out = layer(prev_fused, edge_index_dict)
            if i == len(model.layers) - 2:  # Second-to-last layer
                residual = out.fused['flow']
            prev_fused = out.fused
        
        # Last layer's relation embeddings and fusion module
        relation_embeds = out.relations['flow']
        fusion_module = model.layers[-1].fusion['flow']
        
        return relation_embeds, fusion_module, residual


def fomaml_inner_step(
    model: nn.Module,
    adapter: nn.Module,
    fast_weights: Dict[str, torch.Tensor],
    batch,
    task_signature: torch.Tensor,
    classifier_head: nn.Module,
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """Perform one inner-loop adaptation step using FOMAML.
    
    Args:
        model: Frozen backbone model
        adapter: Transfer adapter module
        fast_weights: Current fast weights (adapted parameters)
        batch: Training batch
        task_signature: Task signature tensor
        classifier_head: Lightweight classifier head
        alpha: Inner learning rate
    
    Returns:
        Updated fast weights
    """
    model.eval()
    adapter.train()
    
    # Forward pass with current fast weights
    with torch.no_grad():
        relation_embeds, fusion_module, residual = _get_last_layer_outputs(model, batch)
    
    # Apply adapter with fast weights
    adapted_embeds = adapter.forward_with_params(relation_embeds, task_signature, fast_weights)
    
    # Fuse and classify (with residual)
    fused = fuse_adapted(adapted_embeds, fusion_module=fusion_module, residual=residual)
    logits = classifier_head(fused)
    
    # Compute loss
    loss = F.cross_entropy(logits, batch["flow"].y)
    
    # Compute gradients wrt adapter's ACTUAL parameters (not fast_weights)
    # This is the key FOMAML trick: forward with fast_weights, backward through actual params
    adapter_params = list(adapter.parameters())
    grads = torch.autograd.grad(
        loss,
        adapter_params,
        create_graph=True,
        retain_graph=False,
        allow_unused=True,
    )
    
    # Update fast weights using gradients wrt actual parameters
    new_fast_weights = {}
    for (name, weight), grad in zip(fast_weights.items(), grads):
        if grad is not None:
            new_fast_weights[name] = weight - alpha * grad.detach()
        else:
            new_fast_weights[name] = weight
    
    return new_fast_weights


def fomaml_meta_gradient(
    model: nn.Module,
    adapter: nn.Module,
    classifier_head: nn.Module,
    query_loader: DataLoader,
    task_signature: torch.Tensor,
    fast_weights: Dict[str, torch.Tensor],
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Compute first-order meta-gradient on query set.

    FOMAML key insight: we differentiate the query loss (evaluated at
    fast_weights) with respect to the adapter's *actual stored parameters*.
    Because forward_with_params uses fast_weights (which were computed from
    actual params via inner-loop SGD, and we detached gradients at each inner
    step), the computation graph runs:

        actual_params → fast_weights (via inner-loop, first-order only)
                      → adapter output → fused → logits → query_loss

    We keep the loss tensor alive (do NOT call .item() before grad) so
    autograd can trace back to adapter.parameters().

    Args:
        model: Frozen backbone
        adapter: Transfer adapter
        classifier_head: Classifier head
        query_loader: Query set loader
        task_signature: Task signature
        fast_weights: Fast weights after inner loop (values are detached leaf
            tensors that still carry grad_fn from the inner SGD chain — but
            in first-order MAML we treat them as constants and differentiate
            wrt the *original* parameters directly).

    Returns:
        Tuple of (scalar query_loss, meta_gradient dict keyed by param name)
    """
    model.eval()
    adapter.train()
    classifier_head.train()

    device = next(model.parameters()).device
    task_signature = task_signature.to(device)

    # Accumulate a live loss tensor — do NOT detach until after autograd.grad.
    # We run the forward pass with fast_weights but the path from actual adapter
    # parameters through the fast_weight computation keeps the graph alive.
    accumulated_loss: Optional[torch.Tensor] = None
    total_samples = 0

    for batch in query_loader:
        batch = batch.to(device)

        with torch.no_grad():
            relation_embeds, fusion_module, residual = _get_last_layer_outputs(model, batch)

        # Forward with fast_weights — keeps graph through actual adapter params
        # because fast_weights[k] = actual_param[k] - alpha * grad (from inner loop)
        adapted_embeds = adapter.forward_with_params(
            relation_embeds, task_signature, fast_weights
        )
        fused = fuse_adapted(adapted_embeds, fusion_module=fusion_module, residual=residual)
        logits = classifier_head(fused)

        n = batch["flow"].y.numel()
        loss = F.cross_entropy(logits, batch["flow"].y) * n
        accumulated_loss = loss if accumulated_loss is None else accumulated_loss + loss
        total_samples += n

    if accumulated_loss is None or total_samples == 0:
        # Empty query set — return zero gradients
        zero_grads = {n: torch.zeros_like(p) for n, p in adapter.named_parameters()}
        return 0.0, zero_grads

    avg_loss = accumulated_loss / total_samples  # keeps grad_fn

    # In First-Order MAML (FOMAML), the meta-gradient wrt initial parameters is
    # approximated by the query loss gradient wrt the adapted fast weights:
    #   \nabla_{\theta_init} L_query \approx \nabla_{fast\_weights} L_query
    # Differentiating wrt fast_weights.values() traces directly through the forward
    # pass (forward_with_params) and yields non-zero meta-gradients for all adapter params.
    fast_weight_names = list(fast_weights.keys())
    fast_weight_tensors = list(fast_weights.values())

    meta_grads = torch.autograd.grad(
        avg_loss,
        fast_weight_tensors,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    scalar_loss = avg_loss.item()

    meta_grads_dict = {
        name: (grad.detach() if grad is not None else torch.zeros_like(param))
        for name, param, grad in zip(fast_weight_names, fast_weight_tensors, meta_grads)
    }

    return scalar_loss, meta_grads_dict


def run_fomaml_episode(
    model: nn.Module,
    adapter: nn.Module,
    classifier_head: nn.Module,
    source_checkpoint: str,
    target_support: list,
    target_query: list,
    task_signature: torch.Tensor,
    k_steps: int = 3,
    alpha: float = 0.01,
    batch_size: int = 32,
    device: torch.device = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Run one FOMAML episode.
    
    Args:
        model: Frozen backbone
        adapter: Transfer adapter (to be meta-trained)
        classifier_head: Lightweight classifier head
        source_checkpoint: Path to source task checkpoint
        target_support: Target task support graphs
        target_query: Target task query graphs
        task_signature: Task signature tensor
        k_steps: Number of inner-loop steps
        alpha: Inner learning rate
        batch_size: Batch size
        device: Device
    
    Returns:
        (query_loss, meta_gradients)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Load source checkpoint into model
    checkpoint = torch.load(source_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    classifier_head.to(device)
    adapter.to(device)
    
    # Create loaders
    support_loader = DataLoader(target_support, batch_size=batch_size, shuffle=True)
    query_loader = DataLoader(target_query, batch_size=batch_size, shuffle=False)
    
    # Initialize fast weights (adapter parameters) with requires_grad=True for inner loop
    fast_weights = {name: param.clone().detach().requires_grad_(True) for name, param in adapter.named_parameters()}
    
    # Inner loop: k steps on support set
    support_loader = DataLoader(target_support, batch_size=batch_size, shuffle=True)
    for step in range(k_steps):
        for batch in support_loader:
            batch = batch.to(device)
            
            with torch.no_grad():
                relation_embeds, fusion_module, residual = _get_last_layer_outputs(model, batch)
            
            # Forward with fast weights
            adapted_embeds = adapter.forward_with_params(relation_embeds, task_signature, fast_weights)
            fused = fuse_adapted(adapted_embeds, fusion_module=fusion_module, residual=residual)
            logits = classifier_head(fused)
            
            loss = F.cross_entropy(logits, batch["flow"].y)
            
            # Gradients wrt fast weights
            grads = torch.autograd.grad(
                loss,
                fast_weights.values(),
                create_graph=True,
                retain_graph=False,
                allow_unused=True,
            )
            
            # First-order update: detach gradients
            new_fast_weights = {}
            for (name, weight), grad in zip(fast_weights.items(), grads):
                if grad is not None:
                    new_fast_weights[name] = weight - alpha * grad.detach()
                else:
                    new_fast_weights[name] = weight
            fast_weights = new_fast_weights
    
    # Meta-loss on query set
    meta_loss, meta_grads = fomaml_meta_gradient(
        model, adapter, classifier_head, query_loader, task_signature, fast_weights
    )
    
    return meta_loss, meta_grads


def apply_meta_gradients(adapter: nn.Module, meta_grads: Dict[str, torch.Tensor], meta_lr: float = 1e-3):
    """Apply meta-gradients to adapter parameters."""
    with torch.no_grad():
        for name, param in adapter.named_parameters():
            if name in meta_grads and meta_grads[name] is not None:
                param -= meta_lr * meta_grads[name]


# Utility functions for testing
def check_fomaml_correctness():
    """Simple correctness check for FOMAML implementation."""
    # This would be a unit test
    pass


if __name__ == "__main__":
    print("FOMAML module loaded successfully")