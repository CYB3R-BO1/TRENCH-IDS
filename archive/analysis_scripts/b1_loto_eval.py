#!/usr/bin/env python3
"""B1 LOTO Evaluation using pre-trained checkpoints.

Evaluates the strong replay baseline checkpoints on held-out next tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.train import load_split
from trench_ids.labels import canonical_classes
from trench_ids.cl.evaluate import compute_metrics


def evaluate_checkpoint_on_task(
    checkpoint_path: Path,
    target_task: int,
    graphs_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Load a checkpoint and evaluate on target task's test split."""
    label_names = canonical_classes()
    model, classifier, checkpoint = load_checkpoint(checkpoint_path, Path("data/graphs"), device)
    model.eval()
    classifier.eval()
    
    target_graphs = load_split(Path("data/graphs"), target_task, "test")
    result = predict(model, classifier, target_graphs, device, 8, canonical_classes())
    metrics = compute_metrics(result["y_true"], result["y_pred"], canonical_classes())
    return metrics


def run_b1_loto(run_dir: Path, seed: int) -> dict:
    """Evaluate all checkpoints on their next task (B1 LOTO)."""
    device = resolve_device("cuda")
    
    transitions = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    results = {}
    
    for src_task, tgt_task in transitions:
        checkpoint_path = Path(f"runs/strong_replay_baseline/checkpoint_task_{src_task}.pt")
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n=== Seed 42 | T{src_task} -> T{tgt_task} ===")
        
        metrics = evaluate_checkpoint_on_task(
            checkpoint_path=checkpoint_path,
            target_task=tgt_task,
            graphs_dir=Path("data/graphs"),
            device=resolve_device("cuda"),
            batch_size=8,
        )
        
        key = f"T{src_task}->T{tgt_task}"
        results[key] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")
    
    return results


if __name__ == "__main__":
    import json
    from pathlib import Path
    from trench_ids.cl.device import resolve_device
    
    all_results = {}
    
    # Run with seed 42 (the strong replay baseline was trained with seed 42)
    print(f"\n{'='*50}")
    print(f"SEED 42")
    print(f"{'='*50}")
    results = {}
    transitions = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    
    for src_task, tgt_task in transitions:
        checkpoint_path = Path(f"runs/strong_replay_baseline/checkpoint_task_{src_task}.pt")
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n=== Seed 42 | T{src_task} -> T{tgt_task} ===")
        
        metrics = evaluate_checkpoint_on_task(
            checkpoint_path=Path(f"runs/strong_replay_baseline/checkpoint_task_{src_task}.pt"),
            target_task=src_task + 1,
            graphs_dir=Path("data/graphs"),
            device=resolve_device("cuda"),
            batch_size=8,
        )
        
        key = f"T{src_task}->T{tgt_task}"
        results[key] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")
    
    all_results[f"seed_42"] = results
    
    # Save results
    out_dir = Path("runs/b1_loto")
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs/b1_loto/b1_results.json")).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to runs/b1_loto/b1_results.json")

if __name__ == "__main__":
    import json
    from pathlib import Path
    from trench_ids.cl.device import resolve_device
    from trench_ids.cl.inference import load_checkpoint, predict
    from trench_ids.cl.train import load_split
    from trench_ids.labels import canonical_classes
    from trench_ids.cl.evaluate import compute_metrics
    
    all_results = {}
    transitions = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    
    results = {}
    for src_task, tgt_task in transitions:
        checkpoint_path = Path(f"runs/strong_replay_baseline/checkpoint_task_{src_task}.pt")
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        print(f"\n=== Seed 42 | T{src_task} -> T{tgt_task} ===")
        
        metrics = evaluate_checkpoint_on_task(
            checkpoint_path=Path(f"runs/strong_replay_baseline/checkpoint_task_{src_task}.pt"),
            target_task=src_task + 1,
            graphs_dir=Path("data/graphs"),
            device=resolve_device("cuda"),
            batch_size=8,
        )
        
        key = f"T{src_task}->T{tgt_task}"
        results[key] = metrics
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1-macro: {metrics['f1_macro']:.4f}")
    
    all_results[f"seed_42"] = results
    
    out_dir = Path("runs/b1_loto")
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs/b1_loto/b1_results.json")).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to runs/b1_loto/b1_results.json")