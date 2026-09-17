"""Data collection for transfer predictor: extract T_r, S_r, D_r, G_r from all sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.cl.drift import boundary_drift, load_mean_transferability
from trench_ids.cl.ewc import FLOW_RELATIONS
from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split, sample_graphs
from trench_ids.labels import NUM_TASKS
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab, vocab_fingerprint


def extract_tr_from_intervention(
    intervention_dir: Path,
    seeds: list[int] = [42, 1],
) -> list[dict[str, Any]]:
    """Extract T_r = AULC_normal - AULC_reset from intervention results."""
    records = []
    for seed in seeds:
        seed_dir = intervention_dir / f"seed{seed}"
        if not seed_dir.exists():
            continue
        for transition_dir in sorted(seed_dir.iterdir()):
            if not transition_dir.is_dir():
                continue
            task_str = transition_dir.name
            # Find normal results - could be "normal" or "normal_1", "normal_2", etc.
            normal_path = None
            for normal_subdir in sorted(transition_dir.glob("normal*")):
                if normal_subdir.is_dir():
                    candidate = normal_subdir / "results.json"
                    if candidate.exists():
                        normal_path = candidate
                        break
            if not normal_path:
                continue
            normal_results = json.loads(normal_path.read_text())
            aulc_normal = normal_results.get("aulc_macro_f1", 0.0)
            for relation in FLOW_RELATIONS:
                reset_path = transition_dir / f"reset_{relation}" / "results.json"
                if not reset_path.exists():
                    continue
                reset_results = json.loads(reset_path.read_text())
                aulc_reset = reset_results.get("aulc_macro_f1", 0.0)
                T_r = aulc_normal - aulc_reset
                records.append({
                    "seed": seed,
                    "transition": task_str,
                    "relation": relation,
                    "T_r": T_r,
                    "aulc_normal": aulc_normal,
                    "aulc_reset": aulc_reset,
                })
    return records


def extract_sr_from_transferability(
    run_dir: Path,
) -> dict[str, dict[int, dict[str, float]]]:
    """Extract S_r per task boundary from transferability_task_*.json files.
    
    Returns: {transition -> {task -> {relation -> S_r}}}
    """
    from trench_ids.constants import FLOW_RELATIONS
    sr_by_boundary = {}
    for path in sorted(run_dir.glob("transferability_task_*.json")):
        task = int(path.stem.split("_")[-1])
        if task >= NUM_TASKS:
            continue
        data = json.loads(path.read_text())
        # Use aggregate_transferability_scores to get mean S_r per relation
        from trench_ids.cl.transferability import aggregate_transferability_scores
        s_r_dict = aggregate_transferability_scores(data, FLOW_RELATIONS)
        s_r = {k: float(v) for k, v in s_r_dict.items()}
        transition = f"t{task}_t{task+1}"
        if transition not in sr_by_boundary:
            sr_by_boundary[transition] = {}
        sr_by_boundary[transition][task] = s_r
    return sr_by_boundary


def run_drift_all_boundaries(
    run_dir: Path,
    graphs_dir: Path,
    device: torch.device,
    seeds: list[int] = [0],
    max_graphs: int = 60,
    batch_size: int = 8,
) -> dict[str, dict[str, float]]:
    """Load pre-computed drift from drift_report.json."""
    results = {}
    drift_report_path = run_dir / "drift_report.json"
    if not drift_report_path.exists():
        return results
    report = json.loads(drift_report_path.read_text())
    boundaries = report.get("boundaries", {})
    for task in range(1, NUM_TASKS):
        key = f"{task}->{task + 1}"
        if key not in boundaries:
            continue
        boundary = boundaries[key]
        # The drift report was computed with seed=0 (or the seed used in drift.py)
        # We'll use it for all seeds since drift is deterministic given the checkpoint
        for seed in seeds:
            transition = f"t{task}_t{task+1}"
            result_key = f"{transition}_seed{seed}"
            drift_dict = {}
            for k, v in boundary.items():
                if k.startswith("drift_") and k != "drift_fused":
                    relation = k.replace("drift_", "")
                    drift_dict[relation] = v
            results[result_key] = drift_dict
    return results


def compute_gradient_alignment(
    run_dir: Path,
    graphs_dir: Path,
    device: torch.device,
    seeds: list[int] = [42, 1],
    max_graphs: int = 60,
    batch_size: int = 8,
) -> dict[str, dict[str, float]]:
    """Load pre-computed gradient alignment from gradient_alignment output."""
    results = {}
    # Try to load from pre-computed files
    grad_alignment_dirs = {
        42: Path("runs/gradient_alignment_seed42"),
        1: Path("runs/gradient_alignment_seed1"),
    }
    
    for seed in seeds:
        grad_dir = grad_alignment_dirs.get(seed)
        if not grad_dir or not grad_dir.exists():
            continue
        comparison_path = grad_dir / "predictor_comparison.json"
        if not comparison_path.exists():
            continue
        data = json.loads(comparison_path.read_text())
        for dp in data.get("data_points", []):
            task = int(dp["transition"].split("->")[0])
            transition = f"t{task}_t{task+1}"
            key = f"{transition}_seed{seed}"
            if key not in results:
                results[key] = {}
            results[key][dp["relation"]] = dp["G_r"]
    return results


# Original compute_gradient_alignment function kept for reference but not used
def _compute_gradient_alignment_original(
    run_dir: Path,
    graphs_dir: Path,
    device: torch.device,
    seeds: list[int] = [42, 1],
    max_graphs: int = 60,
    batch_size: int = 8,
) -> dict[str, dict[str, float]]:
    """Compute gradient alignment G_r = cos(g_r^old, g_r^new) per boundary.
    
    Measures the cosine similarity between gradients of relation-specific
    parameters on old task vs new task data.
    """
    results = {}
    
    for task in range(1, NUM_TASKS):
        earlier_path = run_dir / f"checkpoint_task_{task}.pt"
        later_path = run_dir / f"checkpoint_task_{task + 1}.pt"
        if not (earlier_path.exists() and later_path.exists()):
            continue
            
        # Load model architecture from earlier checkpoint
        checkpoint = torch.load(earlier_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        
        for seed in seeds:
            earlier_model, earlier_classifier, _ = load_checkpoint(earlier_path, graphs_dir, device)
            later_model, later_classifier, _ = load_checkpoint(later_path, graphs_dir, device)
            
            # Get gradients on old task data
            old_graphs = sample_graphs(
                load_split(graphs_dir, task, "test"),
                max_graphs,
                seed=f"{seed}:grad:old:task_{task}",
            )
            # Get gradients on new task data
            new_graphs = sample_graphs(
                load_split(graphs_dir, task + 1, "train"),
                max_graphs,
                seed=f"{seed}:grad:new:task_{task+1}",
            )
            
            relation_grads_old = compute_relation_gradients(
                earlier_model, earlier_classifier, old_graphs, device, batch_size
            )
            relation_grads_new = compute_relation_gradients(
                later_model, later_classifier, new_graphs, device, batch_size
            )
            
            transition = f"t{task}_t{task+1}"
            key = f"{transition}_seed{seed}"
            g_r = {}
            for relation in FLOW_RELATIONS:
                if relation in relation_grads_old and relation in relation_grads_new:
                    g_old = relation_grads_old[relation]
                    g_new = relation_grads_new[relation]
                    if g_old.numel() > 0 and g_new.numel() > 0:
                        cos_sim = F.cosine_similarity(
                            g_old.flatten().unsqueeze(0),
                            g_new.flatten().unsqueeze(0)
                        ).item()
                        g_r[relation] = cos_sim
            results[key] = g_r
    return results


def compute_relation_gradients(
    model: RelationSpecificHeteroGNN,
    classifier: torch.nn.Module,
    graphs: list,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Compute average gradient per relation-specific module."""
    model.train()
    classifier.train()
    grad_accum = {r: [] for r in FLOW_RELATIONS}
    
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        model.zero_grad()
        classifier.zero_grad()
        output = model(batch)
        logits = classifier(output.fused["flow"])
        loss = F.cross_entropy(logits, batch["flow"].y)
        loss.backward()
        
        for layer_idx, layer in enumerate(model.layers):
            conv = layer.conv
            for edge_key, module in conv.rel_lins.items():
                for relation in FLOW_RELATIONS:
                    if f"__{relation}__flow" in edge_key:
                        for param in module.parameters():
                            if param.grad is not None:
                                grad_accum[relation].append(param.grad.clone().flatten())
            for edge_key, module in conv.combine_lins.items():
                for relation in FLOW_RELATIONS:
                    if f"__{relation}__flow" in edge_key:
                        for param in module.parameters():
                            if param.grad is not None:
                                grad_accum[relation].append(param.grad.clone().flatten())
    
    result = {}
    for relation in FLOW_RELATIONS:
        if grad_accum[relation]:
            result[relation] = torch.cat(grad_accum[relation]).mean(dim=0)
    return result


def build_dataset(
    intervention_dir: Path = Path("runs/transfer_intervention"),
    replay_runs: list[Path] = None,
    graphs_dir: Path = Path("data/graphs"),
    device: str = "auto",
    out_dir: Path = Path("runs/transfer_predictor"),
) -> dict[str, Any]:
    """Build the complete transfer predictor dataset.
    
    Returns consolidated dataset with T_r, S_r, D_r, G_r for all
    transition/relation/seed combinations.
    """
    if replay_runs is None:
        replay_runs = [
            Path("runs/gnn_3layer_residual_s42"),
            Path("runs/gnn_3layer_residual_s1"),
            Path("runs/gnn_3layer_residual_s2"),
        ]
    
    device = resolve_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Extract T_r (ground truth) - 50 observations
    print("[data] Extracting T_r from intervention results...")
    tr_records = extract_tr_from_intervention(intervention_dir)
    print(f"[data] Found {len(tr_records)} T_r observations")
    
    # 2. Extract S_r from each replay run
    print("[data] Extracting S_r from transferability files...")
    sr_all = {}
    for run_dir in replay_runs:
        if run_dir.exists():
            sr_all[run_dir.name] = extract_sr_from_transferability(run_dir)
    
    # 3. Run drift (D_r) on all replay runs
    print("[data] Computing D_r (drift) on all boundaries...")
    drift_all = {}
    for run_dir in replay_runs:
        if run_dir.exists():
            drift_all[run_dir.name] = run_drift_all_boundaries(
                run_dir, graphs_dir, device, seeds=[42, 1]
            )
    
    # 4. Compute gradient alignment (G_r) on all replay runs
    print("[data] Computing G_r (gradient alignment) on all boundaries...")
    grad_all = {}
    for run_dir in replay_runs:
        if run_dir.exists():
            grad_all[run_dir.name] = compute_gradient_alignment(
                run_dir, graphs_dir, device
            )
    
    # 5. Consolidate into single dataset
    dataset = []
    for tr in tr_records:
        transition = tr["transition"]
        relation = tr["relation"]
        seed = tr["seed"]
        
        # Find matching S_r from the corresponding seed's replay run
        # Map seed 42 -> gnn_3layer_residual_s42, seed 1 -> gnn_3layer_residual_s1, etc.
        seed_to_run = {42: "gnn_3layer_residual_s42", 1: "gnn_3layer_residual_s1", 2: "gnn_3layer_residual_s2"}
        run_name = seed_to_run.get(seed)
        
        s_r = None
        if run_name and run_name in sr_all:
            sr_data = sr_all[run_name].get(transition, {})
            task = int(transition.split("_")[0][1:])
            s_r = sr_data.get(task, {}).get(relation)
        
        # Find D_r
        d_r = None
        if run_name and run_name in drift_all:
            key = f"{transition}_seed{seed}"
            d_r = drift_all[run_name].get(key, {}).get(relation)
        
        # Find G_r
        g_r = None
        if run_name and run_name in grad_all:
            key = f"{transition}_seed{seed}"
            g_r = grad_all[run_name].get(key, {}).get(relation)
        
        dataset.append({
            "seed": seed,
            "transition": transition,
            "task": int(transition.split("_")[0][1:]),
            "relation": relation,
            "T_r": tr["T_r"],
            "aulc_normal": tr["aulc_normal"],
            "aulc_reset": tr["aulc_reset"],
            "S_r": s_r,
            "D_r": d_r,
            "G_r": g_r,
        })
    
    # Save consolidated dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(dataset)
    df.to_csv(out_dir / "dataset.csv", index=False)
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"[data] Saved {len(dataset)} records to {out_dir}/dataset.csv")
    return {"dataset": dataset, "sr_all": sr_all, "drift_all": drift_all, "grad_all": grad_all}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build transfer predictor dataset")
    parser.add_argument("--intervention-dir", type=Path, default=Path("runs/transfer_intervention"))
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/transfer_predictor"))
    args = parser.parse_args()
    
    build_dataset(
        intervention_dir=args.intervention_dir,
        graphs_dir=args.graphs_dir,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()