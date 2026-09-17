"""Transfer Intervention Experiment: Paired counterfactual continuation to measure
functional forward transfer from relation-specific knowledge.

This experiment tests whether preserving relation-specific parameters from previous
tasks helps learning the next task (forward transfer).

Run:
    python -m trench_ids.cl.transfer_intervention \
        --checkpoint runs/gnn_3layer_residual_s42/checkpoint_task_3.pt \
        --task 4 \
        --mode normal \
        --seed 42 \
        --out-dir runs/transfer_intervention/seed42/t3_t4/normal

    python -m trench_ids.cl.transfer_intervention \
        --checkpoint runs/gnn_3layer_residual_s42/checkpoint_task_3.pt \
        --task 4 \
        --mode reset_relation \
        --relation targeted_by \
        --seed 42 \
        --out-dir runs/transfer_intervention/seed42/t3_t4/reset_targeted_by

    python -m trench_ids.cl.transfer_intervention \
        --checkpoint runs/gnn_3layer_residual_s42/checkpoint_task_3.pt \
        --task 4 \
        --mode reset_random_structured \
        --seed 42 \
        --out-dir runs/transfer_intervention/seed42/t3_t4/random_1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.cl.evaluate import compute_metrics
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.train import (
    build_replay_augmented_set,
    load_split,
    sample_graphs,
    seed_everything,
    split_warmup_and_full_loss_epochs,
)
from trench_ids.cl.stability import (
    StabilityRegularizer,
    extract_relation_parameters,
    compute_parameter_change,
    StabilityConfig,
)
from trench_ids.constants import FLOW_RELATIONS
from trench_ids.labels import attack_classes_for_task, canonical_classes
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab, vocab_fingerprint


@contextmanager
def temporary_seed(seed: int):
    """Temporarily set all RNG states to a deterministic seed, then restore."""
    # Save current states
    torch_cpu_state = torch.get_rng_state()
    torch_cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    
    try:
        # Set temporary seed
        seed_everything(seed)
        yield
    finally:
        # Restore original states
        torch.set_rng_state(torch_cpu_state)
        if torch_cuda_states:
            torch.cuda.set_rng_state_all(torch_cuda_states)
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def get_relation_modules(model: RelationSpecificHeteroGNN, relation: str) -> list[tuple[nn.Module, str]]:
    """Get the exact nn.Linear modules for a relation's rel_lins and combine_lins
    across all layers. Returns list of (module, param_name_prefix)."""
    modules = []
    for layer_idx, layer in enumerate(model.layers):
        conv = layer.conv
        # rel_lins
        rel_lin_key = f"{relation}"  # Will match the edge key pattern
        for edge_key, module in conv.rel_lins.items():
            if f"__{relation}__flow" in edge_key:
                modules.append((module, f"layer{layer_idx}.rel_lins.{edge_key}"))
        # combine_lins
        for edge_key, module in conv.combine_lins.items():
            if f"__{relation}__flow" in edge_key:
                modules.append((module, f"layer{layer_idx}.combine_lins.{edge_key}"))
    return modules


def reinitialize_relation_modules(modules: list[tuple[nn.Module, str]], seed: int) -> None:
    """Reinitialize the exact relation-specific modules using their original
    initialization scheme, within a temporary deterministic RNG context."""
    with temporary_seed(seed):
        for module, _ in modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


def get_structured_random_modules(
    model: RelationSpecificHeteroGNN, 
    exclude_relation: str, 
    seed: int
) -> list[tuple[nn.Module, str]]:
    """Select equivalent relation modules from OTHER FLOW relations,
    matching the same layer structure."""
    all_modules = []
    for relation in FLOW_RELATIONS:
        if relation == exclude_relation:
            continue
        all_modules.extend(get_relation_modules(model, relation))
    
    # We need exactly the same number of modules as the target relation
    target_modules = get_relation_modules(model, exclude_relation)
    target_count = len(target_modules)
    
    rng = random.Random(seed)
    selected = rng.sample(all_modules, target_count)
    return selected


def zero_optimizer_moments(optimizer: torch.optim.Optimizer, param_ids: set[int]) -> None:
    """Zero Adam momentum and variance buffers for the specified parameter IDs."""
    state_dict = optimizer.state_dict()
    for group in state_dict["state"].values():
        for param_id, state in group.items():
            if param_id in param_ids:
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()


def save_full_training_state(
    checkpoint_path: Path,
    run_dir: Path,
    task: int,
    out_path: Path,
) -> None:
    """Save complete training state for exact restoration."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the original run config to get replay buffer path
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    
    # We need to reconstruct the replay buffer from the run directory
    # For the replay baseline, the buffer was built incrementally
    # We'll save the replay buffer state from the run directory
    replay_buffer_path = run_dir / "replay_buffer.pt"
    if replay_buffer_path.exists():
        replay_buffer = torch.load(replay_buffer_path, weights_only=False)
    else:
        replay_buffer = []
    
    # Save RNG states
    torch_cpu_state = torch.get_rng_state()
    torch_cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    
    torch.save({
        "replay_buffer": replay_buffer,
        "rng_states": {
            "torch_cpu": torch_cpu_state,
            "torch_cuda": torch_cuda_states,
            "numpy": numpy_state,
            "python": python_state,
        },
        "config": config,
    }, out_path)


def load_full_training_state(
    checkpoint_path: Path,
    run_dir: Path,
    task: int,
    in_path: Path,
    device: torch.device,
) -> dict:
    """Restore complete training state for exact continuation."""
    state = torch.load(in_path, map_location="cpu", weights_only=False)
    return state


def continue_training(
    model: RelationSpecificHeteroGNN,
    classifier: nn.Linear,
    optimizer: torch.optim.Optimizer,
    replay_buffer: list,
    task_graphs: list,
    next_task_graphs: list,
    device: torch.device,
    graphs_dir: Path,
    task: int,
    epochs_per_task: int,
    warmup_epochs: int,
    batch_size: int,
    learning_rate: float,
    replay_fraction: float,
    replay_selection: str,
    track_per_epoch: bool = True,
    stability_regularizer: StabilityRegularizer = None,
) -> dict:
    """Continue training from task t to t+1, recording per-epoch metrics.
    
    Args:
        stability_regularizer: Optional StabilityRegularizer to constrain parameter changes.
    """
    model.train()
    classifier.train()
    
    # Build replay-augmented training set for next task
    training_set = build_replay_augmented_set(
        next_task_graphs, 
        replay_buffer, 
        replay_fraction
    )
    
    warmup_epochs_local, full_loss_epochs = split_warmup_and_full_loss_epochs(
        epochs_per_task, warmup_epochs
    )
    total_epochs = warmup_epochs_local + full_loss_epochs
    
    loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    
    label_names = canonical_classes()
    
    # Test graphs for next task and old tasks
    next_task_test = load_split(graphs_dir, task + 1, "test")
    old_tasks = list(range(1, task + 1))
    old_task_tests = {t: load_split(graphs_dir, t, "test") for t in old_tasks}
    
    per_epoch_metrics = []
    
    for epoch in range(total_epochs):
        # Training
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            logits = classifier(output.fused["flow"])
            loss = torch.nn.functional.cross_entropy(logits, batch["flow"].y)
            
            # Add stability regularization if provided
            if stability_regularizer is not None:
                stab_loss = stability_regularizer(model)
                loss = loss + stab_loss
            
            loss.backward()
            optimizer.step()
        
        # Per-epoch evaluation
        if track_per_epoch:
            # Evaluate on next task (t+1)
            pred = predict(model, classifier, next_task_test, device, batch_size, label_names)
            metrics = compute_metrics(pred["y_true"], pred["y_pred"], label_names)
            
            # Also evaluate on old tasks for retention diagnostic
            old_task_metrics = {}
            for old_task in old_tasks:
                pred_old = predict(model, classifier, old_task_tests[old_task], device, batch_size, label_names)
                old_metrics = compute_metrics(pred_old["y_true"], pred_old["y_pred"], label_names)
                old_task_metrics[f"task_{old_task}"] = old_metrics
            
            per_epoch_metrics.append({
                "epoch": epoch,
                "next_task": metrics,
                "old_tasks": old_task_metrics,
            })
    
    return {"per_epoch": per_epoch_metrics}


def compute_aulc(per_epoch_metrics: list[dict], metric_key: str = "f1_macro") -> float:
    """Compute Area Under Learning Curve using trapezoidal rule."""
    values = [m["next_task"][metric_key] for m in per_epoch_metrics]
    if len(values) < 2:
        return 0.0
    aulc = 0.0
    for i in range(len(values) - 1):
        aulc += (values[i] + values[i + 1]) / 2.0
    return aulc


def run_intervention(
    checkpoint_path: Path,
    run_dir: Path,
    graphs_dir: Path,
    task: int,
    mode: str,
    relation: str | None,
    seed: int,
    out_dir: Path,
    device: torch.device,
    epochs_per_task: int = 5,
    warmup_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    replay_fraction: float = 0.3,
    replay_selection: str = "uniform",
    stability_mode: str = "none",
    stability_lambda: float = 1.0,
    stability_oracle_weights: dict[str, float] = None,
) -> dict:
    """Run a single intervention experiment.
    
    Args:
        stability_mode: "none", "uniform", "targeted_by", "oracle"
        stability_lambda: Weight for stability regularization
        stability_oracle_weights: For oracle mode, relation -> weight mapping
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint and config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    # Load model and classifier
    model, classifier, _ = load_checkpoint(checkpoint_path, graphs_dir, device)
    
    # Load replay buffer from run directory - use the buffer state AFTER task t
    # (i.e., before starting task t+1). This is replay_buffer_task_t.pt
    replay_buffer_path = run_dir / f"replay_buffer_task_{task}.pt"
    if replay_buffer_path.exists():
        replay_buffer = torch.load(replay_buffer_path, weights_only=False)
    else:
        replay_buffer = []
    
    # Restore optimizer state from the checkpoint's task
    # Original training includes importance_mlp parameters in optimizer
    # We need to recreate the same parameter groups
    importance_mlp = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    ).to(device)
    trainable_params = (
        list(model.parameters()) + list(classifier.parameters()) + list(importance_mlp.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    optimizer_state_path = run_dir / f"optimizer_task_{task}.pt"
    if optimizer_state_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
    
    # Get next task graphs
    next_task_graphs = load_split(graphs_dir, task + 1, "train")
    task_graphs = load_split(graphs_dir, task, "train")
    
    # Create stability regularizer if requested
    stability_regularizer = None
    if stability_mode != "none" and stability_lambda > 0:
        stability_config = StabilityConfig(
            enabled=True,
            lambda_stability=stability_lambda,
            mode=stability_mode,
            oracle_weights=stability_oracle_weights,
        )
        stability_regularizer = StabilityRegularizer(model, stability_config, device)
    
    # Apply intervention
    intervention_info = {"mode": mode, "stability_mode": stability_mode, "stability_lambda": stability_lambda}
    
    if mode == "reset_relation" and relation:
        modules = get_relation_modules(model, relation)
        param_ids = {id(p) for module, _ in modules for p in module.parameters()}
        
        # Reinitialize
        reinitialize_relation_modules(modules, seed)
        
        # Zero optimizer moments for intervened parameters
        zero_optimizer_moments(optimizer, param_ids)
        
        intervention_info["relation"] = relation
        intervention_info["modules_reset"] = len(modules)
        intervention_info["params_reset"] = sum(p.numel() for module, _ in modules for p in module.parameters())
    
    elif mode == "reset_random_structured":
        modules = get_structured_random_modules(model, relation or "targeted_by", seed)
        param_ids = {id(p) for module, _ in modules for p in module.parameters()}
        
        reinitialize_relation_modules(modules, seed)
        zero_optimizer_moments(optimizer, param_ids)
        
        intervention_info["relation"] = relation
        intervention_info["modules_reset"] = len(modules)
        intervention_info["params_reset"] = sum(p.numel() for module, _ in modules for p in module.parameters())
        intervention_info["control_type"] = "structured_random"
    
    elif mode != "normal":
        raise ValueError(f"Unknown mode: {mode}")
    
    # Run continuation with stability regularizer
    results = continue_training(
        model, classifier, optimizer, replay_buffer,
        task_graphs, next_task_graphs, device,
        graphs_dir, task,
        epochs_per_task, warmup_epochs, batch_size, learning_rate,
        replay_fraction, replay_selection, track_per_epoch=True,
        stability_regularizer=stability_regularizer,
    )
    
    # Compute AULC
    aulc_f1 = compute_aulc(results["per_epoch"], "f1_macro")
    aulc_acc = compute_aulc(results["per_epoch"], "accuracy")
    epoch1_f1 = results["per_epoch"][0]["next_task"]["f1_macro"] if results["per_epoch"] else 0.0
    early_avg_f1 = np.mean([m["next_task"]["f1_macro"] for m in results["per_epoch"][:3]]) if len(results["per_epoch"]) >= 3 else 0.0
    final_f1 = results["per_epoch"][-1]["next_task"]["f1_macro"] if results["per_epoch"] else 0.0
    
    summary = {
        "intervention": intervention_info,
        "aulc_macro_f1": aulc_f1,
        "aulc_accuracy": aulc_acc,
        "epoch1_macro_f1": epoch1_f1,
        "early_avg_macro_f1": early_avg_f1,
        "final_macro_f1": final_f1,
        "per_epoch": results["per_epoch"],
    }
    
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer Intervention: paired counterfactual continuation experiment."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--task", type=int, required=True, help="Current task t (continuing to t+1)")
    parser.add_argument("--mode", choices=["normal", "reset_relation", "reset_random_structured"], required=True)
    parser.add_argument("--relation", type=str, help="Relation name for reset_relation mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    # Training parameters (must match original run)
    parser.add_argument("--epochs-per-task", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--replay-fraction", type=float, default=0.3)
    parser.add_argument("--replay-selection", type=str, default="uniform")
    # Stability regularization parameters
    parser.add_argument("--stability-mode", choices=["none", "uniform", "targeted_by", "oracle"], default="none",
                        help="Stability regularization mode")
    parser.add_argument("--stability-lambda", type=float, default=1.0,
                        help="Stability regularization weight")
    parser.add_argument("--stability-oracle-weights", type=str, default=None,
                        help="JSON string for oracle weights, e.g., '{\"targeted_by\": 1.0, \"originates\": 0.5}'")
    args = parser.parse_args()
    
    device = resolve_device(args.device)
    
    if args.mode == "reset_relation" and not args.relation:
        raise ValueError("--relation required for reset_relation mode")
    
    # For structured random control, we need a reference relation to match structure
    if args.mode == "reset_random_structured" and not args.relation:
        args.relation = "targeted_by"  # default reference
    
    # Parse oracle weights if provided
    oracle_weights = None
    if args.stability_oracle_weights:
        import json
        oracle_weights = json.loads(args.stability_oracle_weights)
    
    summary = run_intervention(
        args.checkpoint,
        args.run_dir,
        args.graphs_dir,
        args.task,
        args.mode,
        args.relation,
        args.seed,
        args.out_dir,
        device,
        epochs_per_task=args.epochs_per_task,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        replay_fraction=args.replay_fraction,
        replay_selection=args.replay_selection,
        stability_mode=args.stability_mode,
        stability_lambda=args.stability_lambda,
        stability_oracle_weights=oracle_weights,
    )
    
    print(f"[done] {args.mode} {args.relation or ''}: AULC F1={summary['aulc_macro_f1']:.4f}, Epoch1 F1={summary['epoch1_macro_f1']:.4f}, Final F1={summary['final_macro_f1']:.4f}")


if __name__ == "__main__":
    main()