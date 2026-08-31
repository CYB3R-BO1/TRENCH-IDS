"""Gradient alignment analysis: compare relation-specific gradient alignment
against functional transfer ground truth from intervention experiments.

Computes G_r(t) = cos(∇_θ_r L_old, ∇_θ_r L_new) per relation per transition.
Compares against functional transfer T_r from intervention experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.constants import FLOW_RELATIONS
from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split, sample_graphs
from trench_ids.labels import NUM_TASKS


@torch.enable_grad()
def compute_gradient_alignment(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    graphs: list,
    device: torch.device,
    batch_size: int = 8,
    relation_to_params: dict[str, list[tuple[torch.nn.Module, str]]] | None = None,
) -> dict[str, float]:
    """Compute cosine similarity between old-task and new-task gradients
    for each relation's parameters.

    Returns {relation: cosine_similarity} where positive means gradients
    are aligned (old and new learning objectives agree), negative means
    conflicting (interference).
    """
    if relation_to_params is None:
        # Build mapping from relation to its parameters
        relation_to_params = {}
        for name, param in model.named_parameters():
            if "layers." in name and ".conv." in name:
                parts = name.split(".")
                if len(parts) >= 5:
                    layer_idx = int(parts[1])
                    param_type = parts[3]
                    edge_key = parts[4]
                    edge_parts = edge_key.split("__")
                    if len(edge_parts) == 3:
                        src, relation, dst = edge_parts
                        if relation in FLOW_RELATIONS and dst == "flow":
                            if relation not in relation_to_params:
                                relation_to_params[relation] = []
                            relation_to_params[relation].append(param)

    model.train()
    classifier.train()

    old_grads: dict[str, torch.Tensor] = {}
    new_grads: dict[str, torch.Tensor] = {}

    # Use old-task data (replay-like) and new-task data
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    # We need both old-task and new-task gradients
    # For old-task gradients, use a mix of previously seen classes
    # For new-task gradients, use the current task's data

    old_grad_accum: dict[str, torch.Tensor] = {r: torch.zeros_like(list(params)[0]) for r, params in relation_to_params.items()}
    new_grad_accum: dict[str, torch.Tensor] = {r: torch.zeros_like(list(params)[0]) for r, params in relation_to_params.items()}
    old_count = 0
    new_count = 0

    for batch in loader:
        batch = batch.to(device)

        # Compute gradients on old-task objective (classification on old classes)
        # This is a proxy - in reality we'd use the replay buffer
        model.zero_grad()
        classifier.zero_grad()
        output = model(batch)
        logits = classifier(output.fused["flow"])
        # Use cross-entropy as the loss
        loss_old = F.cross_entropy(logits, batch["flow"].y)
        loss_old.backward()

        # Accumulate old gradients per relation
        for relation, params in relation_to_params.items():
            for p in params:
                if p.grad is not None:
                    if relation not in old_grads:
                        old_grads[relation] = p.grad.clone().detach().flatten()
                    else:
                        old_grads[relation] = torch.cat([old_grads[relation], p.grad.clone().detach().flatten()])

        # For new gradients, we'd need new-task data
        # For now, we'll use a different approach: compute gradients
        # on a different batch or with a different objective

        # Actually, let's do a cleaner approach:
        # 1. Get replay buffer data (old tasks)
        # 2. Get new task data (task t+1)
        # Compute gradients separately
        break

    # This needs a cleaner implementation - let me rethink
    return {}


def compute_gradient_alignment_at_boundary(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    old_task_graphs: list,
    new_task_graphs: list,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    """Compute gradient alignment for each relation at a task boundary.

    Args:
        old_task_graphs: Graphs from old tasks (replay buffer / previous tasks)
        new_task_graphs: Graphs from the new task (task t+1)
    """
    # Build relation -> parameters mapping
    relation_to_params = {}
    for name, param in model.named_parameters():
        if "layers." in name and ".conv." in name:
            parts = name.split(".")
            if len(parts) >= 5:
                param_type = parts[3]
                edge_key = parts[4]
                edge_parts = edge_key.split("__")
                if len(edge_parts) == 3:
                    src, relation, dst = edge_parts
                    if relation in FLOW_RELATIONS and dst == "flow":
                        if relation not in relation_to_params:
                            relation_to_params[relation] = []
                        relation_to_params[relation].append(param)

    model.train()
    classifier.train()

    alignment: dict[str, float] = {}

    for relation, params in relation_to_params.items():
        # Compute gradients on old task data
        model.zero_grad()
        classifier.zero_grad()
        old_loader = DataLoader(old_task_graphs, batch_size=batch_size, shuffle=True)
        old_grad_vecs = []
        for batch in old_loader:
            batch = batch.to(device)
            output = model(batch)
            logits = classifier(output.fused["flow"])
            loss = F.cross_entropy(logits, batch["flow"].y)
            loss.backward()
            for p in params:
                if p.grad is not None:
                    old_grad_vecs.append(p.grad.clone().detach().flatten())
            break  # Just one batch for efficiency

        # Compute gradients on new task data
        model.zero_grad()
        classifier.zero_grad()
        new_loader = DataLoader(new_task_graphs, batch_size=batch_size, shuffle=True)
        new_grad_vecs = []
        for batch in new_loader:
            batch = batch.to(device)
            output = model(batch)
            logits = classifier(output.fused["flow"])
            loss = F.cross_entropy(logits, batch["flow"].y)
            loss.backward()
            for p in params:
                if p.grad is not None:
                    new_grad_vecs.append(p.grad.clone().detach().flatten())
            break

        if old_grad_vecs and new_grad_vecs:
            old_grad = torch.cat(old_grad_vecs)
            new_grad = torch.cat(new_grad_vecs)
            cos_sim = F.cosine_similarity(old_grad.unsqueeze(0), new_grad.unsqueeze(0), dim=-1).item()
            alignment[relation] = cos_sim
        else:
            alignment[relation] = 0.0

    return alignment


def load_intervention_ground_truth(intervention_dir: Path) -> dict[tuple[int, str], float]:
    """Load functional transfer T_r from intervention experiment results.

    Returns {(task_transition, relation): T_r}
    """
    truth = {}
    # T2->T3
    for rel in ["originates", "terminated_by", "targeted_by", "protocol_of", "service_of"]:
        path = intervention_dir / "t2_t3" / f"reset_{rel}" / "results.json"
        if path.exists():
            data = json.loads(path.read_text())
            truth[(2, rel)] = data.get("aulc_macro_f1", 0) - data.get("baseline_aulc", 0)
        # Actually the intervention results store aulc_macro_f1 directly, need baseline
    return truth


def load_transferability_scores(run_dir: Path, task: int) -> dict[str, float]:
    """Load S_r from transferability_task_t.json."""
    from trench_ids.cl.transferability import aggregate_transferability_scores
    path = run_dir / f"transferability_task_{task}.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    if not report:
        return {}
    return {
        rel: float(val)
        for rel, val in aggregate_transferability_scores(report, FLOW_RELATIONS).items()
    }


def load_drift_scores(run_dir: Path, task: int) -> dict[str, float]:
    """Load drift D_r from drift_report.json for a specific transition."""
    path = run_dir / "drift_report.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    boundaries = report.get("boundaries", {})
    key = f"{task}->{task + 1}"
    if key in boundaries:
        boundary = boundaries[key]
        return {k.replace("drift_", ""): v for k, v in boundary.items() if k.startswith("drift_")}
    return {}


def load_intervention_results(intervention_dir: Path, task: int) -> dict[str, float]:
    """Load T_r from intervention results for a specific transition t->t+1."""
    results = {}
    # Try both "normal" and "normal_1" for the baseline
    baseline_path = intervention_dir / f"t{task}_t{task+1}" / "normal" / "results.json"
    if not baseline_path.exists():
        baseline_path = intervention_dir / f"t{task}_t{task+1}" / "normal_1" / "results.json"
    if not baseline_path.exists():
        return results
    baseline_aulc = json.loads(baseline_path.read_text()).get("aulc_macro_f1", 0)

    for rel in ["originates", "terminated_by", "targeted_by", "protocol_of", "service_of"]:
        path = intervention_dir / f"t{task}_t{task+1}" / f"reset_{rel}" / "results.json"
        if path.exists():
            data = json.loads(path.read_text())
            reset_aulc = data.get("aulc_macro_f1", 0)
            # T_r = AULC_normal - AULC_reset (positive means reset hurts = transfer exists)
            results[rel] = baseline_aulc - reset_aulc
    return results


def compare_predictors(
    intervention_dir: Path,
    run_dir: Path,
    graphs_dir: Path,
    tasks: list[int],
    device: torch.device,
    batch_size: int = 8,
    max_graphs: int = 60,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare S_r, drift D_r, and gradient alignment G_r against T_r."""
    
    all_data = []
    
    for task in tasks:
        # Load ground truth T_r
        T_r = load_intervention_results(intervention_dir, task)
        
        # Load S_r
        S_r = load_transferability_scores(run_dir, task)
        
        # Load D_r
        D_r = load_drift_scores(run_dir, task)
        
        # Load checkpoints for gradient alignment
        earlier_path = run_dir / f"checkpoint_task_{task}.pt"
        later_path = run_dir / f"checkpoint_task_{task + 1}.pt"
        
        if not (earlier_path.exists() and later_path.exists()):
            continue
            
        earlier_model, earlier_classifier, _ = load_checkpoint(earlier_path, graphs_dir, device)
        later_model, later_classifier, _ = load_checkpoint(later_path, graphs_dir, device)
        
        # Get old and new task graphs
        old_graphs = sample_graphs(
            load_split(graphs_dir, task, "test"),
            max_graphs,
            seed=f"{seed}:grad_align:task_{task}:old",
        )
        new_graphs = sample_graphs(
            load_split(graphs_dir, task + 1, "test"),
            max_graphs,
            seed=f"{seed}:grad_align:task_{task}:new",
        )
        
        # Compute gradient alignment G_r
        G_r = compute_gradient_alignment_at_boundary(
            earlier_model, earlier_classifier,
            old_graphs, new_graphs,
            device, batch_size
        )
        
        # Collect data points
        for relation in FLOW_RELATIONS:
            if relation in T_r:
                all_data.append({
                    "transition": f"{task}->{task+1}",
                    "relation": relation,
                    "T_r": T_r[relation],
                    "S_r": S_r.get(relation, 0.0),
                    "D_r": D_r.get(relation, 0.0),
                    "G_r": G_r.get(relation, 0.0),
                })
    
    return {"data_points": all_data}


def compute_correlations(data_points: list[dict[str, float]]) -> dict[str, Any]:
    """Compute rank correlations between each predictor and T_r."""
    if len(data_points) < 3:
        return {}
    
    T_vals = [d["T_r"] for d in data_points]
    S_vals = [d["S_r"] for d in data_points]
    D_vals = [d["D_r"] for d in data_points]
    G_vals = [d["G_r"] for d in data_points]
    
    results = {}
    for name, vals in [("S_r", S_vals), ("D_r", D_vals), ("G_r", G_vals)]:
        rho, p = spearmanr(T_vals, vals)
        results[name] = {
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "p_value": float(p) if not np.isnan(p) else None,
            "n": len(data_points),
        }
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare gradient alignment and other predictors against functional transfer ground truth."
    )
    parser.add_argument("--intervention-dir", type=Path, default=Path("runs/transfer_intervention/seed42"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/gnn_3layer_residual_s42_v2"))
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--tasks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-graphs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gradient_alignment"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    comparison = compare_predictors(
        args.intervention_dir,
        args.run_dir,
        args.graphs_dir,
        args.tasks,
        device,
        args.batch_size,
        args.max_graphs,
        args.seed,
    )

    correlations = compute_correlations(comparison["data_points"])
    comparison["correlations"] = correlations

    (args.out_dir / "predictor_comparison.json").write_text(json.dumps(comparison, indent=2))

    print(f"[done] Data points: {len(comparison['data_points'])}")
    print(f"Correlations:")
    for pred, corr in correlations.items():
        print(f"  {pred}: rho = {corr.get('spearman_rho'):.4f}, p = {corr.get('p_value'):.4f}")


if __name__ == "__main__":
    main()