"""Meta-Transfer Training Loop for TCTRL.

Implements the B0 POC training loop with four conditions:
1. Baseline (no adapter)
2. Random adapter (capacity control)
3. Meta-trained TCTRL (proposed method)
4. Oracle adapter (upper bound)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.cl.episode_generator import EpisodeGenerator, generate_b0_meta_data
from trench_ids.cl.fomaml import run_fomaml_episode, apply_meta_gradients, fomaml_inner_step
from trench_ids.cl.task_signature import TaskSignatureExtractor, SignatureStats, extract_task_signature
from trench_ids.cl.train import load_split, sample_graphs, build_replay_augmented_set, split_warmup_and_full_loss_epochs
from trench_ids.constants import FLOW_RELATIONS
from trench_ids.model.transfer_adapter import (
    TransferAdapter, RandomTransferAdapter, LightweightClassifierHead, fuse_adapted
)
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.evaluate import compute_metrics
from trench_ids.labels import canonical_classes, NUM_TASKS
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab, vocab_fingerprint


def train_b0_poc(
    checkpoint_dir: Path,
    graphs_dir: Path,
    meta_train_episodes: List[Dict],
    meta_val_episodes: List[Dict],
    task: int = 3,  # T3->T4
    seed: int = 42,
    device: str = "auto",
    out_dir: Path = Path("runs/meta_transfer/B0_poc"),
    # Meta-training hyperparameters
    meta_epochs: int = 10,
    meta_lr: float = 1e-3,
    inner_lr: float = 0.01,
    k_steps: int = 1,
    batch_size: int = 32,
    meta_batch_size: int = 8,  # episodes per meta-batch
    # Adapter hyperparameters
    adapter_dim: int = 32,
    hidden_dim: int = 64,
    # Evaluation
    eval_batch_size: int = 32,
    eval_k_steps: int = 3,
    inner_lr_eval: float = 0.01,
    # Replay
    replay_fraction: float = 0.3,
    replay_selection: str = "uniform",
) -> Dict:
    """Run B0 POC: Train and evaluate TCTRL on T3->T4 transition.

    Args:
        checkpoint_dir: Path to checkpoint_task_3.pt.
        graphs_dir: Path to graphs directory.
        meta_train_episodes: List of meta-train episode dicts.
        meta_val_episodes: List of meta-val episode dicts.
        task: Source task number (3 for T3->T4).
        seed: Random seed.
        device: Device.
        out_dir: Output directory.
        meta_epochs: Number of meta-training epochs.
        meta_lr: Meta learning rate.
        inner_lr: Inner-loop learning rate.
        k_steps: Number of inner-loop adaptation steps.
        batch_size: Batch size for inner loop.
        meta_batch_size: Episodes per meta-batch.
        adapter_dim: Adapter bottleneck dimension.
        hidden_dim: Hidden dimension.
        eval_batch_size: Evaluation batch size.
        eval_k_steps: Evaluation inner steps.
        inner_lr_eval: Inner LR for evaluation.
        replay_fraction: Replay fraction for CL loop (for retention eval).
        replay_selection: Replay selection strategy.

    Returns:
        Dictionary with results for all four conditions.
    """
    print(f"=== train_b0_poc called: task={task}, meta_epochs={meta_epochs}, max_episodes_override=None ===")
    print(f"  Device: {device}")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Graphs: {graphs_dir}")
    device = resolve_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load frozen backbone
    checkpoint_path = checkpoint_dir / f"checkpoint_task_{task}.pt"
    model, base_classifier, _ = load_checkpoint(checkpoint_path, graphs_dir, device)
    model.eval()
    base_classifier.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in base_classifier.parameters():
        p.requires_grad = False
    
    # Load replay buffer (for retention evaluation)
    replay_buffer_path = checkpoint_dir / f"replay_buffer_task_{task}.pt"
    if replay_buffer_path.exists():
        replay_buffer = torch.load(replay_buffer_path, weights_only=False)
    else:
        replay_buffer = []
    
    # Use the B0 episode support/query graphs when available. The full task
    # train/test splits are much larger than the meta-learning support/query
    # sets and make the smoke test take orders of magnitude longer than the
    # intended 1-episode proof-of-concept run.
    target_task = task + 1
    if meta_train_episodes:
        target_support = meta_train_episodes[0].get("target_support", load_split(graphs_dir, target_task, "train"))
        target_query = (meta_val_episodes[0].get("target_query") if meta_val_episodes else meta_train_episodes[0].get("target_query"))
        if target_query is None:
            target_query = load_split(graphs_dir, target_task, "test")
    else:
        target_support = load_split(graphs_dir, target_task, "train")
        target_query = load_split(graphs_dir, target_task, "test")
    
    # Extract task signature from frozen encoder
    signature_extractor = TaskSignatureExtractor(model, device, batch_size=32)
    # We'll compute stats from meta-train episodes
    # First, extract signatures from meta-train episodes
    meta_train_signatures = []
    for idx, episode in enumerate(meta_train_episodes):
        if idx % 50 == 0:
            print(f"  Extracting signature {idx+1}/{len(meta_train_episodes)}", flush=True)
        sig = extract_task_signature(
            episode['target_support'], model, device,
            stats=None, include_covariance=False, batch_size=32
        )
        meta_train_signatures.append(sig)
    
    print(f"  Extracted {len(meta_train_signatures)} signatures")
    
    # Compute normalization stats
    all_sigs = torch.stack(meta_train_signatures)
    emb_mean = all_sigs[:, :64].mean(0)
    emb_std = all_sigs[:, :64].std(0, unbiased=False).clamp(min=1e-6)  # Use biased std to avoid div-by-zero
    rel_freq = all_sigs[:, 128:133]
    rel_freq_min = rel_freq.min(0).values
    rel_freq_max = torch.maximum(rel_freq.max(0).values, rel_freq_min + 1e-6)
    node_counts = all_sigs[:, 133:137]
    node_min = node_counts.min(0).values
    node_max = torch.maximum(node_counts.max(0).values, node_min + 1e-6)
    
    stats = SignatureStats(
        emb_mean=emb_mean,
        emb_std=emb_std,
        rel_freq_min=rel_freq_min,
        rel_freq_max=rel_freq_max,
        node_count_min=node_min,
        node_count_max=node_max,
    )
    
    print(f"  Computed stats: emb_mean={emb_mean.shape}, emb_std={emb_std.shape}")
    
    # Normalize meta-val signatures too
    meta_val_signatures = []
    for episode in meta_val_episodes:
        sig = extract_task_signature(
            episode['target_support'], model, device,
            stats=None, include_covariance=False, batch_size=32
        )
        meta_val_signatures.append(sig)
    
    # Normalize T3->T4 test signature
    test_signature = extract_task_signature(target_support, model, device, stats=None)
    test_signature_norm = normalize_signature(test_signature, stats)
    
    print("Signatures extracted and normalized")
    conditions = {
        'baseline': {'use_adapter': False, 'adapter_type': None},
        'random_adapter': {'use_adapter': True, 'adapter_type': 'random'},
        'meta_trained': {'use_adapter': True, 'adapter_type': 'meta'},
        'oracle': {'use_adapter': True, 'adapter_type': 'oracle'},
    }
    
    results = {}
    
    # Create optimizer for meta-training (only adapter params)
    def create_adapter(adapter_type: str) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
        """Create adapter based on type."""
        if adapter_type == 'random':
            adapter = RandomTransferAdapter(hidden_dim=hidden_dim, adapter_dim=adapter_dim)
            adapter.to(device)
            return adapter, None
        elif adapter_type == 'meta':
            adapter = TransferAdapter(hidden_dim=hidden_dim, adapter_dim=adapter_dim)
            adapter.to(device)
            optimizer = torch.optim.Adam(adapter.parameters(), lr=meta_lr)
            return adapter, optimizer
        elif adapter_type == 'oracle':
            adapter = TransferAdapter(hidden_dim=hidden_dim, adapter_dim=adapter_dim)
            adapter.to(device)
            # Will create optimizer with adapter + classifier_head in caller
            return adapter, None
        else:
            return None, None
    
    # Training loop for each condition
    for cond_name, cond_config in conditions.items():
        print(f"\n=== Training condition: {cond_name} ===")
        
        if cond_name == 'baseline':
            results[cond_name] = evaluate_condition(
                model, base_classifier, None, target_support, target_query,
                test_signature_norm, device, eval_batch_size, eval_k_steps,
                inner_lr_eval, replay_buffer, replay_fraction, replay_selection,
                seed, out_dir / cond_name, graphs_dir=graphs_dir
            )
            continue
        
        # Create adapter
        adapter, optimizer = create_adapter(cond_config['adapter_type'])
        # All conditions use 64-dim fused embedding (after attention fusion)
        classifier_head = LightweightClassifierHead(
            input_dim=hidden_dim,  # 64, after attention fusion
            num_classes=11
        ).to(device)
        
        if cond_name == 'oracle':
            # Train oracle on target support with labels - optimizer includes adapter + head
            oracle_optimizer = torch.optim.Adam(
                list(adapter.parameters()) + list(classifier_head.parameters()), lr=1e-3
            )
            train_oracle_adapter(
                model, adapter, classifier_head, target_support, target_query,
                test_signature_norm, device, batch_size, inner_lr_eval,
                20, oracle_optimizer, out_dir / cond_name
            )
            # Oracle already trained - just evaluate, don't re-train
            results[cond_name] = evaluate_condition(
                model, base_classifier, adapter, target_support, target_query,
                test_signature_norm, device, eval_batch_size, eval_k_steps,
                inner_lr_eval, replay_buffer, replay_fraction, replay_selection,
                seed, out_dir / cond_name, graphs_dir=graphs_dir,
                classifier_head=classifier_head,
                freeze_adapter=True, freeze_head=True
            )
        else:
            # Meta-training for TCTRL and random adapter
            if cond_name == 'meta_trained':
                meta_train_tctrl(
                    model, adapter, classifier_head, meta_train_episodes,
                    meta_val_episodes, stats, device, meta_epochs, meta_lr,
                    inner_lr, k_steps, batch_size, meta_batch_size,
                    optimizer, out_dir / cond_name
                )
            
            # Evaluate
            results[cond_name] = evaluate_condition(
                model, base_classifier, adapter, target_support, target_query,
                test_signature_norm, device, eval_batch_size, eval_k_steps,
                inner_lr_eval, replay_buffer, replay_fraction, replay_selection,
                seed, out_dir / cond_name, graphs_dir=graphs_dir,
                classifier_head=classifier_head
            )
    
    # Save results - only numeric top-level metrics, exclude per-epoch details
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'seed': seed,
        'task': task,
        'conditions': {
            k: {
                m: float(v) for m, v in r.items() 
                if m != 'per_epoch' and isinstance(v, (int, float, bool))
            }
            for k, r in results.items()
        },
        'hyperparameters': {
            'meta_epochs': meta_epochs,
            'meta_lr': meta_lr,
            'inner_lr': inner_lr,
            'k_steps': k_steps,
            'adapter_dim': adapter_dim,
            'hidden_dim': hidden_dim,
        }
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    
    # Print comparison
    print("\n=== B0 POC Results ===")
    for cond, res in results.items():
        aulc = res.get('aulc_1_3', 0)
        print(f"  {cond}: AULC(1-3) = {aulc:.4f}")
    
    return results


def meta_train_tctrl(
    model: nn.Module,
    adapter: TransferAdapter,
    classifier_head: LightweightClassifierHead,
    meta_train_episodes: List[Dict],
    meta_val_episodes: List[Dict],
    stats: SignatureStats,
    device: torch.device,
    meta_epochs: int,
    meta_lr: float,
    inner_lr: float,
    k_steps: int,
    batch_size: int,
    meta_batch_size: int,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
) -> Dict:
    """Meta-train the TCTRL adapter using FOMAML."""
    
    model.eval()
    best_val_loss = float('inf')
    best_state = None
    
    # Normalize meta-val signatures
    val_signatures = []
    for ep in meta_val_episodes:
        sig = ep.get('target_signature_norm') or extract_task_signature(
            ep['target_support'], model, device,
            stats=stats
        )
        val_signatures.append(sig)
    
    for epoch in range(meta_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle episodes
        indices = list(range(len(meta_train_episodes)))
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), meta_batch_size):
            batch_indices = indices[i:i+meta_batch_size]
            
            batch_loss = 0.0
            for idx in batch_indices:
                episode = meta_train_episodes[idx]
                
                # Get source checkpoint and target graphs
                source_checkpoint = episode.get('source_checkpoint')
                target_support = episode['target_support']
                target_query = episode['target_query']
                
                # Get task signature
                task_sig = episode.get('target_signature_norm')
                if task_sig is None:
                    task_sig = extract_task_signature(
                        target_support, model, device, stats=stats
                    )
                
                # Run FOMAML episode
                meta_loss, meta_grads = run_fomaml_episode(
                    model,
                    adapter,
                    classifier_head,
                    source_checkpoint,
                    target_support,
                    target_query,
                    task_sig,
                    k_steps=k_steps,
                    alpha=inner_lr,
                    batch_size=batch_size,
                    device=device,
                )
                
                # Apply meta-gradients
                apply_meta_gradients(adapter, meta_grads, meta_lr)
                batch_loss += meta_loss
            
            epoch_loss += batch_loss / max(1, len(batch_indices))
            n_batches += 1
        
        avg_loss = epoch_loss / max(1, n_batches)
        
        # Validation
        val_loss = evaluate_meta_validation(model, adapter, None, meta_val_episodes, stats, device)
        
        print(f"  Epoch {epoch+1}/{meta_epochs}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
    
    # Restore best
    if best_state:
        adapter.load_state_dict(best_state)
    
    # Save adapter
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), out_dir / "adapter_best.pt")
    
    return {'final_train_loss': epoch_loss, 'best_val_loss': best_val_loss}


def predict_with_adapter(
    model: nn.Module,
    adapter: nn.Module,
    head: nn.Module,
    graphs: list,
    task_signature: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict:
    """Run inference through the adapter path: backbone → adapter → fuse → head.

    This is the correct evaluation path for all adapter conditions (random,
    meta-trained, oracle).  The standard ``predict()`` from inference.py calls
    ``model(batch)`` then ``backbone_classifier(output.fused)`` which bypasses
    the adapter entirely.

    Args:
        model: Frozen backbone.
        adapter: Trained (or random/oracle) transfer adapter.
        head: LightweightClassifierHead fitted during evaluation.
        graphs: List of HeteroData graphs to evaluate on.
        task_signature: Normalised task signature tensor.
        device: Compute device.
        batch_size: DataLoader batch size.

    Returns:
        Dict with y_true, y_pred, y_prob lists.
    """
    model.eval()
    adapter.eval()
    head.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []

    task_sig = task_signature.to(device)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            relation_embeds, fusion_module, residual = _get_last_layer_outputs(model, batch)
            adapted = adapter(relation_embeds, task_sig)
            fused = fuse_adapted(adapted, fusion_module=fusion_module, residual=residual)
            logits = head(fused)
            probs = F.softmax(logits, dim=-1)
            y_true.extend(batch["flow"].y.tolist())
            y_pred.extend(logits.argmax(dim=-1).tolist())
            y_prob.extend(probs.tolist())

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def evaluate_meta_validation(
    model: nn.Module,
    adapter: TransferAdapter,
    classifier_head: Optional[nn.Module],
    meta_val_episodes: List[Dict],
    stats: SignatureStats,
    device: torch.device,
    k_steps: int = 1,
    inner_lr: float = 0.01,
    batch_size: int = 32,
) -> float:
    """Evaluate on meta-validation episodes using k=1 inner adaptation.

    For each val episode: run k inner-loop steps on support, then measure
    query loss with the adapted fast_weights.  Returns mean query loss across
    all episodes — used for best-model selection during meta-training.

    Args:
        model: Frozen backbone.
        adapter: Current meta-trained adapter (weights to evaluate).
        classifier_head: Classifier head (if None, creates a fresh one).
        meta_val_episodes: List of meta-val episode dicts.
        stats: Normalization stats for signature extraction.
        device: Compute device.
        k_steps: Inner-loop adaptation steps (default 1 for fast validation).
        inner_lr: Inner-loop learning rate.
        batch_size: Batch size.

    Returns:
        Mean query loss (scalar float) — lower is better.
    """
    from trench_ids.cl.fomaml import run_fomaml_episode

    model.eval()
    adapter.eval()

    if not meta_val_episodes:
        return 0.0

    # Use a fresh head per episode (same as training) so we don't accumulate
    # adaptation from one val episode into the next.
    total_loss = 0.0
    n_episodes = 0

    for episode in meta_val_episodes:
        target_support = episode.get("target_support", [])
        target_query = episode.get("target_query", [])
        source_checkpoint = episode.get("source_checkpoint")

        if not target_support or not target_query or not source_checkpoint:
            continue

        # Extract task signature
        task_sig = episode.get("target_signature_norm")
        if task_sig is None:
            task_sig = extract_task_signature(
                target_support, model, device, stats=stats
            )

        # Create a fresh head and run FOMAML episode (support adapt → query loss)
        val_head = LightweightClassifierHead(input_dim=64, num_classes=11).to(device)
        try:
            ep_loss, _ = run_fomaml_episode(
                model,
                adapter,
                val_head,
                source_checkpoint,
                target_support,
                target_query,
                task_sig,
                k_steps=k_steps,
                alpha=inner_lr,
                batch_size=batch_size,
                device=device,
            )
        except Exception:
            continue

        total_loss += ep_loss
        n_episodes += 1

    return total_loss / n_episodes if n_episodes > 0 else 0.0


def _get_last_layer_outputs(model: nn.Module, batch):
    """Extract last layer's relation embeddings and previous layer's fused output (for residual)."""
    model.eval()
    with torch.no_grad():
        x_dict = model.encoders(batch)
        edge_index_dict = batch.edge_index_dict
        
        prev_fused = x_dict
        for i, layer in enumerate(model.layers):
            out = layer(prev_fused, edge_index_dict)
            if i == len(model.layers) - 2:
                residual = out.fused['flow']
            prev_fused = out.fused
        
        relation_embeds = out.relations['flow']
        fusion_module = model.layers[-1].fusion['flow']
        
        return relation_embeds, fusion_module, residual


def train_oracle_adapter(
    model: nn.Module,
    adapter: TransferAdapter,
    classifier_head: LightweightClassifierHead,
    target_support: list,
    target_query: list,
    task_signature: torch.Tensor,
    device: torch.device,
    batch_size: int,
    inner_lr: float,
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    out_dir: Path,
) -> None:
    """Train oracle adapter on target support with labels."""
    model.eval()
    adapter.train()
    classifier_head.train()
    
    support_loader = DataLoader(target_support, batch_size=batch_size, shuffle=True)
    query_loader = DataLoader(target_query, batch_size=batch_size, shuffle=False)
    
    for epoch in range(n_epochs):
        for batch in support_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                relation_embeds, fusion_module, residual = _get_last_layer_outputs(model, batch)
            
            adapted = adapter(relation_embeds, task_signature)
            # Use model's fusion module (last layer's flow fusion) + residual
            fused = fuse_adapted(adapted, fusion_module=fusion_module, residual=residual)
            logits = classifier_head(fused)
            loss = F.cross_entropy(logits, batch["flow"].y)
            loss.backward()
            optimizer.step()
        
        # Evaluate on query
        model.eval()
        adapter.eval()
        classifier_head.eval()
        query_loss = 0
        for batch in query_loader:
            batch = batch.to(device)
            with torch.no_grad():
                relation_embeds, fusion_module, residual = _get_last_layer_outputs(model, batch)
                adapted = adapter(relation_embeds, task_signature)
                fused = fuse_adapted(adapted, fusion_module=fusion_module, residual=residual)
                logits = classifier_head(fused)
                query_loss += F.cross_entropy(logits, batch["flow"].y).item() * batch["flow"].y.numel()
        
        avg_loss = query_loss / len(target_query)
        if epoch % 5 == 0:
            print(f"  Oracle epoch {epoch+1}: query_loss={avg_loss:.4f}")


def evaluate_condition(
    model: nn.Module,
    base_classifier: nn.Module,
    adapter: Optional[TransferAdapter],
    target_support: list,
    target_query: list,
    task_signature: torch.Tensor,
    device: torch.device,
    batch_size: int,
    k_steps: int,
    inner_lr: float,
    replay_buffer: list,
    replay_fraction: float,
    replay_selection: str,
    seed: int,
    out_dir: Path,
    graphs_dir: Path = Path("data/graphs"),
    classifier_head: Optional[LightweightClassifierHead] = None,
    freeze_adapter: bool = False,
    freeze_head: bool = False,
) -> Dict:
    """Evaluate a condition on target task.
    
    Returns metrics including AULC, epoch-1 F1, forgetting, etc.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create lightweight classifier head if not provided
    # All conditions use 64-dim fused embedding (after attention fusion)
    head = classifier_head or LightweightClassifierHead(64, 11).to(device)
    
    # Build replay-augmented target support
    if replay_buffer and replay_fraction > 0:
        training_set = build_replay_augmented_set(target_support, replay_buffer, replay_fraction)
    else:
        training_set = target_support
    
    # Loaders
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True)
    query_loader = DataLoader(target_query, batch_size=batch_size, shuffle=False)
    
    # Optimizer for inner adaptation
    adapt_params = []
    if head is not None and not freeze_head:
        adapt_params.extend(head.parameters())
    if adapter and not freeze_adapter:
        adapt_params.extend(adapter.parameters())
    
    # If nothing to optimize, just evaluate without training
    if len(adapt_params) == 0:
        optimizer = None
    else:
        optimizer = torch.optim.SGD(adapt_params, lr=inner_lr)

    per_epoch_metrics = []
    
    for epoch in range(k_steps + 5):  # Warmup + full
        epoch_loss = 0
        n_samples = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            
            if adapter:
                with torch.no_grad():
                    relation_embeds, fusion_module, residual = _get_last_layer_outputs(model, batch)
                task_sig = task_signature.to(device)
                if freeze_adapter:
                    with torch.no_grad():
                        adapted = adapter(relation_embeds, task_sig)
                else:
                    adapted = adapter(relation_embeds, task_sig)
                fused = fuse_adapted(adapted, fusion_module=fusion_module, residual=residual)
            else:
                with torch.no_grad():
                    output = model(batch)
                fused = output.fused["flow"]
            
            logits = head(fused)
            loss = F.cross_entropy(logits, batch["flow"].y)
            
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item() * batch["flow"].y.numel()
            n_samples += batch["flow"].y.numel()
        
        # Evaluate on query — route through adapter when present
        if adapter is not None:
            # predict_with_adapter: backbone → adapter → fuse_adapted → head
            pred = predict_with_adapter(
                model, adapter, head, target_query, task_signature, device, batch_size
            )
        else:
            # Baseline: backbone → base_classifier (no adapter path)
            pred = predict(model, base_classifier, target_query, device, batch_size, canonical_classes())
        metrics = compute_metrics(pred["y_true"], pred["y_pred"], canonical_classes())

        # Also evaluate on old tasks for forgetting (always via backbone + base_classifier
        # since old tasks have no adapter-trained path)
        old_metrics = {}
        for old_task in range(1, 4):  # T1, T2, T3
            old_test = load_split(graphs_dir, old_task, "test")
            pred_old = predict(model, base_classifier, old_test, device, batch_size, canonical_classes())
            old_metrics[f"task_{old_task}"] = compute_metrics(pred_old["y_true"], pred_old["y_pred"], canonical_classes())
        
        per_epoch_metrics.append({
            "epoch": epoch,
            "loss": epoch_loss / max(1, n_samples),
            "query": metrics,
            "old_tasks": old_metrics,
        })
    
    # Compute AULC (epochs 1-3)
    f1_scores = [m["query"]["f1_macro"] for m in per_epoch_metrics[:3]]
    aulc_1_3 = sum((f1_scores[i] + f1_scores[i+1]) / 2 for i in range(len(f1_scores)-1)) if len(f1_scores) > 1 else f1_scores[0]
    
    # Final metrics
    final_metrics = per_epoch_metrics[-1]["query"]
    
    result = {
        "aulc_1_3": aulc_1_3,
        "epoch_1_f1": per_epoch_metrics[0]["query"]["f1_macro"] if per_epoch_metrics else 0,
        "mean_f1_1_3": np.mean(f1_scores) if f1_scores else 0,
        "final_f1": final_metrics.get("f1_macro", 0),
        "final_acc": final_metrics.get("accuracy", 0),
        "forgetting": compute_forgetting(per_epoch_metrics),
        "per_epoch": per_epoch_metrics,
    }
    
    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))
    
    return result


def compute_forgetting(per_epoch_metrics: List[Dict]) -> float:
    """Compute average forgetting from per-epoch metrics."""
    if not per_epoch_metrics:
        return 0.0
    
    forgetting = 0
    count = 0
    final_epoch = per_epoch_metrics[-1]
    
    for task_name, metrics in final_epoch.get("old_tasks", {}).items():
        # Find peak accuracy for this task
        peak_acc = max(m["old_tasks"].get(task_name, {}).get("accuracy", 0) for m in per_epoch_metrics)
        final_acc = metrics.get("accuracy", 0)
        forgetting += max(0, peak_acc - final_acc)
        count += 1
    
    return forgetting / count if count > 0 else 0.0


def normalize_signature(signature: torch.Tensor, stats: SignatureStats) -> torch.Tensor:
    """Normalize signature using meta-train stats."""
    normalized = signature.clone()
    
    # Embeddings
    normalized[:64] = (normalized[:64] - stats.emb_mean) / stats.emb_std
    normalized[64:128] = (normalized[64:128] - stats.emb_mean) / stats.emb_std
    
    # Relation frequencies
    rel = normalized[128:133]
    normalized[128:133] = (rel - stats.rel_freq_min) / (stats.rel_freq_max - stats.rel_freq_min + 1e-8)
    
    # Node counts
    nc = normalized[133:137]
    normalized[133:137] = (nc - stats.node_count_min) / (stats.node_count_max - stats.node_count_min + 1e-8)
    
    return normalized


if __name__ == "__main__":
    print("Meta-transfer training module loaded successfully")