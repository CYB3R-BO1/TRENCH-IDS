#!/usr/bin/env python3
"""B1 LOTO Evaluation on strong replay baseline.

For each transition T_n -> T_{n+1}:
1. Load checkpoint after task n (3-layer residual + replay)
2. Fine-tune on task n+1 training data for 3 epochs
3. Measure accuracy/F1 at epochs 1, 2, 3
4. Compute AULC(1-3)
"""

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeoDataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.train import load_split
from trench_ids.labels import canonical_classes
from trench_ids.cl.evaluate import compute_metrics


def fine_tune_and_evaluate(
    model: torch.nn.Module,
    classifier: torch.nn.Linear,
    target_task: int,
    device: torch.device,
    fine_tune_epochs: int = 3,
    batch_size: int = 8,
) -> list[dict]:
    """Fine-tune model on target task training data, return per-epoch metrics."""
    
    target_test = load_split(Path("data/graphs"), target_task, "test")
    target_train = list(load_split(Path("data/graphs"), target_task, "train"))
    
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(classifier.parameters()),
        lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    
    per_epoch = []
    
    for epoch in range(3):
        # Fine-tune for one epoch
        model.train()
        
        # Create a fresh loader each epoch with shuffled data
        fine_tune_sample = random.sample(target_train, min(200, len(target_train)))
        loader = GeoDataLoader(fine_tune_sample, batch_size=8, shuffle=True)
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            logits = classifier(output.fused["flow"])
            loss = F.cross_entropy(logits, batch["flow"].y)
            loss.backward()
            optimizer.step()
        
        # Evaluate on test set after this epoch
        model.eval()
        classifier.eval()
        with torch.no_grad():
            result = predict(model, classifier, load_split(Path("data/graphs"), target_task, "test"), device, 8, canonical_classes())
        metrics = compute_metrics(result["y_true"], result["y_pred"], canonical_classes())
        
        per_epoch.append({
            "epoch": epoch + 1,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
        })
    
    return per_epoch


def run_b1_transition(src_task: int, seed: int = 42) -> dict:
    """Run B1 evaluation for one transition T_n -> T_{n+1}."""
    import torch
    
    device = resolve_device("cuda")
    target_task = src_task + 1
    
    checkpoint_path = Path(f"runs/fact_strong/checkpoint_task_{src_task}.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"  T{src_task} -> T{target_task}...")
    
    # Load checkpoint
    model, classifier, _ = load_checkpoint(
        Path(f"runs/strong_replay_baseline/checkpoint_task_{src_task}.pt"),
        Path("data/graphs"),
        resolve_device("cuda")
    )
    
    # Fine-tune and evaluate
    per_epoch = fine_tune_and_evaluate(model, classifier, src_task + 1, resolve_device("cuda"))
    
    return {
        "transition": f"T{src_task}->T{src_task+1}",
        "per_epoch": per_epoch,
        "final": per_epoch[-1] if per_epoch else {}
    }


def run_b1_loto(seed: int = 42) -> dict:
    """Run B1 LOTO for all 5 transitions."""
    import torch
    
    torch.manual_seed(seed)
    random.seed(seed)
    
    transitions = [1, 2, 3, 4, 5]
    results = {}
    
    for src_task in [1, 2, 3, 4, 5]:
        print(f"  T{src_task} -> T{src_task+1}...")
        
        try:
            result = run_b1_transition(src_task, seed)
            results[f"T{src_task}->T{src_task+1}"] = result
            final = result.get("final", {})
            print(f"  Final Acc: {final.get('accuracy', 0):.4f}, F1: {final.get('f1_macro', 0):.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[f"T{src_task}->T{src_task+1}"] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    import json
    import random
    import torch
    
    from trench_ids.cl.device import resolve_device
    from trench_ids.cl.inference import load_checkpoint, predict
    from trench_ids.cl.train import load_split
    from trench_ids.labels import canonical_classes
    from trench_ids.cl.evaluate import compute_metrics
    from torch_geometric.loader import DataLoader as GeoDataLoader
    
    all_results = {}
    
    for seed in [42, 1]:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")
        
        torch.manual_seed(seed)
        random.seed(seed)
        
        results = run_b1_loto(seed)
        all_results[f"seed_{seed}"] = results
    
    # Save results
    out_dir = Path("runs/b1_loto")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "b1_results_fact.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to runs/b1_loto/b1_results_fact.json")

if __name__ == "__main__":
    import json
    import random
    import torch
    
    all_results = {}
    
    for seed in [42, 1]:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")
        
        torch.manual_seed(seed)
        random.seed(seed)
        
        results = run_b1_loto(seed)
        all_results[f"seed_{seed}"] = results
    
    out_dir = Path("runs/b1_loto")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "b1_results_fact.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to runs/b1_loto/b1_results_fact.json")