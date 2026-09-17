"""Mechanistic Analysis: compare relation-specific properties against functional transfer T_r.

This module systematically measures relation-level properties at each task boundary
and compares them against the intervention ground truth T_r.

Goal: discover what mechanism actually causes positive functional forward transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.cl.drift import boundary_drift
from trench_ids.cl.ewc import FLOW_RELATIONS
from trench_ids.cl.inference import load_checkpoint
from trench_ids.cl.train import load_split, sample_graphs
from trench_ids.labels import NUM_TASKS
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab


def load_intervention_ground_truth(intervention_dir: Path) -> dict[tuple[int, str], float]:
    """Load functional transfer T_r from intervention experiment results.
    
    Returns {(transition_task, relation): T_r}
    """
    truth = {}
    for task in range(1, NUM_TASKS):
        transition_dir = intervention_dir / f"t{task}_t{task+1}"
        if not transition_dir.exists():
            continue
        # Find normal results
        normal_path = None
        for normal_subdir in sorted(transition_dir.glob("normal*")):
            if normal_subdir.is_dir():
                candidate = normal_subdir / "results.json"
                if candidate.exists():
                    normal_path = candidate
                    break
        if not normal_path:
            continue
        baseline_aulc = json.loads(normal_path.read_text()).get("aulc_macro_f1", 0)

        for rel in FLOW_RELATIONS:
            path = transition_dir / f"reset_{rel}" / "results.json"
            if path.exists():
                data = json.loads(path.read_text())
                reset_aulc = data.get("aulc_macro_f1", 0)
                truth[(task, rel)] = baseline_aulc - reset_aulc
    return truth


@torch.no_grad()
def compute_representation_geometry(
    model: RelationSpecificHeteroGNN,
    graphs: list,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, dict[str, float]]:
    """Compute representation geometry per relation.
    
    Measures:
    - Intra-class variance (compactness)
    - Inter-class distance (separation)
    - Class separation margin (Fisher discriminant ratio)
    - Normalized separation = inter / intra
    """
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    # Collect embeddings per relation per class
    relation_class_embeddings: dict[str, dict[int, list[torch.Tensor]]] = {
        r: {} for r in FLOW_RELATIONS
    }
    
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        y = batch["flow"].y
        
        for relation in FLOW_RELATIONS:
            emb = output.relations["flow"].get(relation)
            if emb is None:
                continue
            for c in y.unique():
                c = int(c.item())
                mask = (y == c)
                if mask.any():
                    class_emb = emb[mask]
                    if relation not in relation_class_embeddings:
                        relation_class_embeddings[relation] = {}
                    if c not in relation_class_embeddings[relation]:
                        relation_class_embeddings[relation][c] = []
                    relation_class_embeddings[relation][c].append(class_emb.cpu())
    
    results = {}
    for relation in FLOW_RELATIONS:
        class_embs = relation_class_embeddings.get(relation, {})
        if len(class_embs) < 2:
            results[relation] = {"intra_var": 0.0, "inter_dist": 0.0, "separation_ratio": 0.0}
            continue
        
        # Concatenate embeddings per class
        class_means = {}
        class_vars = {}
        for c, embs in class_embs.items():
            all_emb = torch.cat(embs, dim=0)
            class_means[c] = all_emb.mean(dim=0)
            class_vars[c] = all_emb.var(dim=0).mean().item()
        
        classes = list(class_means.keys())
        
        # Intra-class variance (mean across classes)
        intra_var = np.mean(list(class_vars.values()))
        
        # Inter-class distance (mean pairwise distance between class means)
        if len(classes) >= 2:
            means_matrix = torch.stack([class_means[c] for c in classes])
            dists = pdist(means_matrix.numpy(), metric='euclidean')
            inter_dist = float(np.mean(dists))
        else:
            inter_dist = 0.0
        
        # Separation ratio
        separation_ratio = inter_dist / (intra_var + 1e-8)
        
        results[relation] = {
            "intra_var": float(intra_var),
            "inter_dist": inter_dist,
            "separation_ratio": float(separation_ratio),
        }
    
    return results


@torch.no_grad()
def compute_parameter_stability(
    model: RelationSpecificHeteroGNN,
) -> dict[str, dict[str, float]]:
    """Compute parameter statistics per relation module.
    
    Measures:
    - Mean parameter magnitude
    - Parameter count
    - Parameter variance across layers
    """
    results = {}
    for relation in FLOW_RELATIONS:
        rel_params = []
        for layer_idx, layer in enumerate(model.layers):
            conv = layer.conv
            # rel_lins
            for edge_key, module in conv.rel_lins.items():
                if f"__{relation}__flow" in edge_key:
                    for param in module.parameters():
                        rel_params.append(param.detach().flatten())
            # combine_lins
            for edge_key, module in conv.combine_lins.items():
                if f"__{relation}__flow" in edge_key:
                    for param in module.parameters():
                        rel_params.append(param.detach().flatten())
        
        if rel_params:
            all_params = torch.cat(rel_params)
            results[relation] = {
                "param_count": int(all_params.numel()),
                "param_mean": float(all_params.mean()),
                "param_std": float(all_params.std()),
                "param_norm": float(all_params.norm()),
            }
        else:
            results[relation] = {
                "param_count": 0,
                "param_mean": 0.0,
                "param_std": 0.0,
                "param_norm": 0.0,
            }
    return results


def compute_parameter_changes(
    earlier_model: RelationSpecificHeteroGNN,
    later_model: RelationSpecificHeteroGNN,
) -> dict[str, dict[str, float]]:
    """Compute parameter changes per relation between two checkpoints."""
    results = {}
    for relation in FLOW_RELATIONS:
        changes = []
        earlier_params = []
        later_params = []
        
        for layer_idx, (earlier_layer, later_layer) in enumerate(zip(earlier_model.layers, later_model.layers)):
            earlier_conv = earlier_layer.conv
            later_conv = later_layer.conv
            
            # rel_lins
            for edge_key, earlier_module in earlier_conv.rel_lins.items():
                if f"__{relation}__flow" in edge_key:
                    if edge_key in later_conv.rel_lins:
                        later_module = later_conv.rel_lins[edge_key]
                        for ep, lp in zip(earlier_module.parameters(), later_module.parameters()):
                            diff = (lp - ep).detach().flatten()
                            changes.append(diff)
                            earlier_params.append(ep.detach().flatten())
                            later_params.append(lp.detach().flatten())
            
            # combine_lins
            for edge_key, earlier_module in earlier_conv.combine_lins.items():
                if f"__{relation}__flow" in edge_key:
                    if edge_key in later_conv.combine_lins:
                        later_module = later_conv.combine_lins[edge_key]
                        for ep, lp in zip(earlier_module.parameters(), later_module.parameters()):
                            diff = (lp - ep).detach().flatten()
                            changes.append(diff)
                            earlier_params.append(ep.detach().flatten())
                            later_params.append(lp.detach().flatten())
        
        if changes:
            all_changes = torch.cat(changes)
            all_earlier = torch.cat(earlier_params)
            all_later = torch.cat(later_params)
            
            results[relation] = {
                "param_change_norm": float(all_changes.norm()),
                "param_change_mean": float(all_changes.mean()),
                "param_change_std": float(all_changes.std()),
                "relative_change": float(all_changes.norm() / (all_earlier.norm() + 1e-8)),
            }
        else:
            results[relation] = {
                "param_change_norm": 0.0,
                "param_change_mean": 0.0,
                "param_change_std": 0.0,
                "relative_change": 0.0,
            }
    return results


@torch.enable_grad()
def compute_gradient_magnitude(
    model: RelationSpecificHeteroGNN,
    classifier: torch.nn.Module,
    graphs: list,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    """Compute gradient magnitude per relation on given graphs."""
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
                                grad_accum[relation].append(param.grad.detach().flatten())
            for edge_key, module in conv.combine_lins.items():
                for relation in FLOW_RELATIONS:
                    if f"__{relation}__flow" in edge_key:
                        for param in module.parameters():
                            if param.grad is not None:
                                grad_accum[relation].append(param.grad.detach().flatten())
    
    results = {}
    for relation in FLOW_RELATIONS:
        if grad_accum[relation]:
            all_grads = torch.cat(grad_accum[relation])
            results[relation] = {
                "grad_norm": float(all_grads.norm()),
                "grad_mean": float(all_grads.mean()),
                "grad_std": float(all_grads.std()),
            }
        else:
            results[relation] = {
                "grad_norm": 0.0,
                "grad_mean": 0.0,
                "grad_std": 0.0,
            }
    return results


@torch.no_grad()
def compute_attention_contribution(
    model: RelationSpecificHeteroGNN,
    graphs: list,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    """Compute SemanticAttention contribution per relation for Flow nodes."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    attention_sums = {r: 0.0 for r in FLOW_RELATIONS}
    attention_counts = {r: 0 for r in FLOW_RELATIONS}
    
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        
        # Attention weights are in output.attention: {node_type: {relation: beta}}
        if hasattr(output, 'attention') and output.attention:
            flow_attn = output.attention.get('flow', {})
            for relation, weight in flow_attn.items():
                if relation in attention_sums:
                    # weight is a tensor of shape [N] (per-node attention weights)
                    if isinstance(weight, torch.Tensor):
                        attention_sums[relation] += float(weight.mean())
                    else:
                        attention_sums[relation] += float(weight)
                    attention_counts[relation] += 1
    
    results = {}
    for relation in FLOW_RELATIONS:
        if attention_counts[relation] > 0:
            results[relation] = float(attention_sums[relation] / attention_counts[relation])
        else:
            results[relation] = 0.0
    return results


def compute_cross_dataset_similarity(
    model: RelationSpecificHeteroGNN,
    graphs_dir: Path,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, float]:
    """Compute same-class representation similarity across datasets.
    
    For classes that appear in multiple datasets (e.g., DDoS in ToN and CSE),
    measure cosine similarity of their relation-specific embeddings.
    """
    # This requires dataset metadata in the graphs - simplified for now
    # TODO: implement when dataset labels are available in graph data
    return {r: 0.0 for r in FLOW_RELATIONS}


def run_analysis_at_boundary(
    task: int,
    graphs_dir: Path,
    run_dir: Path,
    device: torch.device,
    seed: int = 42,
    max_graphs: int = 60,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run all mechanistic analyses at a specific task boundary t → t+1."""
    
    # Load checkpoints
    earlier_path = run_dir / f"checkpoint_task_{task}.pt"
    later_path = run_dir / f"checkpoint_task_{task + 1}.pt"
    
    if not (earlier_path.exists() and later_path.exists()):
        return {"error": f"Checkpoints not found for task {task}"}
    
    earlier_model, earlier_classifier, _ = load_checkpoint(earlier_path, graphs_dir, device)
    later_model, later_classifier, _ = load_checkpoint(later_path, graphs_dir, device)
    
    # Sample graphs
    old_graphs = sample_graphs(
        load_split(graphs_dir, task, "test"),
        max_graphs,
        seed=f"{seed}:mechanistic:task_{task}",
    )
    new_graphs = sample_graphs(
        load_split(graphs_dir, task + 1, "train"),
        max_graphs,
        seed=f"{seed}:mechanistic:task_{task+1}",
    )
    all_graphs = old_graphs + new_graphs
    
    # Run analyses
    print(f"  [analysis] Computing representation geometry...")
    geometry = compute_representation_geometry(earlier_model, all_graphs, device, batch_size)
    
    print(f"  [analysis] Computing parameter stability...")
    param_stability = compute_parameter_stability(earlier_model)
    
    print(f"  [analysis] Computing parameter changes...")
    param_changes = compute_parameter_changes(earlier_model, later_model)
    
    print(f"  [analysis] Computing gradient magnitude...")
    grad_mag = compute_gradient_magnitude(earlier_model, earlier_classifier, old_graphs, device, batch_size)
    
    print(f"  [analysis] Computing attention contribution...")
    attention = compute_attention_contribution(earlier_model, all_graphs, device, batch_size)
    
    print(f"  [analysis] Computing drift...")
    drift = boundary_drift(
        earlier_model, earlier_classifier, later_model, later_classifier,
        old_graphs, device, batch_size,
    )
    
    # Load functional ground truth
    intervention_dir = Path("runs/transfer_intervention/seed42")  # Default, can be parameterized
    truth = load_intervention_ground_truth(intervention_dir)
    T_r = {rel: truth.get((task, rel), 0.0) for rel in FLOW_RELATIONS}
    
    return {
        "task": task,
        "transition": f"t{task}_t{task+1}",
        "geometry": geometry,
        "param_stability": param_stability,
        "param_changes": param_changes,
        "grad_magnitude": grad_mag,
        "attention": attention,
        "drift": {k: v for k, v in drift.items() if k.startswith("drift_")},
        "T_r": T_r,
    }


def correlate_with_T_r(
    analysis_results: dict[str, Any],
    property_name: str,
    metric_name: str,
) -> dict[str, float]:
    """Correlate a mechanistic property with T_r."""
    relations = FLOW_RELATIONS
    T_vals = [analysis_results["T_r"].get(r, 0.0) for r in relations]
    prop_vals = [analysis_results[property_name].get(r, {}).get(metric_name, 0.0) for r in relations]
    
    if len(set(T_vals)) < 2 or len(set(prop_vals)) < 2:
        return {"spearman_rho": None, "pearson_r": None}
    
    rho, p = spearmanr(T_vals, prop_vals)
    r, p2 = np.corrcoef(T_vals, prop_vals)[0, 1], None
    
    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else None,
        "pearson_r": float(r) if not np.isnan(r) else None,
    }


def run_full_analysis(
    graphs_dir: Path = Path("data/graphs"),
    run_dirs: list[Path] = None,
    seeds: list[int] = [42, 1],
    device: str = "auto",
    out_dir: Path = Path("runs/mechanistic_analysis"),
) -> dict[str, Any]:
    """Run mechanistic analysis across all boundaries and runs."""
    
    if run_dirs is None:
        run_dirs = {
            42: Path("runs/gnn_3layer_residual_s42"),
            1: Path("runs/gnn_3layer_residual_s1"),
            2: Path("runs/gnn_3layer_residual_s2"),
        }
    
    device = resolve_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for seed in seeds:
        run_dir = run_dirs.get(seed)
        if not run_dir or not run_dir.exists():
            print(f"[analysis] Run dir not found for seed {seed}: {run_dir}")
            continue
        
        print(f"[analysis] Running for seed {seed} ({run_dir})...")
        seed_results = {}
        
        for task in range(1, NUM_TASKS):
            print(f"  [analysis] Task boundary {task} -> {task+1}...")
            result = run_analysis_at_boundary(
                task, graphs_dir, run_dir, device, seed=seed
            )
            seed_results[f"t{task}_t{task+1}"] = result
        
        all_results[f"seed{seed}"] = seed_results
    
    # Save
    (out_dir / "mechanistic_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"[analysis] Results saved to {out_dir}/mechanistic_results.json")
    
    return all_results


def print_correlation_summary(results: dict[str, Any]):
    """Print correlation summary for a single transition."""
    # Extract the first seed's results for a specific task
    for seed_key, seed_results in results.items():
        for transition_key, trans_result in seed_results.items():
            if "error" in trans_result:
                continue
            
            print(f"\n{seed_key} - {transition_key}:")
            print(f"  Functional T_r: {trans_result['T_r']}")
            
            # Correlation with geometry
            for metric in ["intra_var", "inter_dist", "separation_ratio"]:
                corr = correlate_with_T_r(trans_result, "geometry", metric)
                print(f"  geometry.{metric}: rho={corr['spearman_rho']:.4f}" if corr['spearman_rho'] else f"  geometry.{metric}: N/A")
            
            # Correlation with parameter changes
            for metric in ["param_change_norm", "relative_change"]:
                corr = correlate_with_T_r(trans_result, "param_changes", metric)
                print(f"  param_changes.{metric}: rho={corr['spearman_rho']:.4f}" if corr['spearman_rho'] else f"  param_changes.{metric}: N/A")
            
            # Correlation with gradient magnitude
            for metric in ["grad_norm"]:
                corr = correlate_with_T_r(trans_result, "grad_magnitude", metric)
                print(f"  grad_magnitude.{metric}: rho={corr['spearman_rho']:.4f}" if corr['spearman_rho'] else f"  grad_magnitude.{metric}: N/A")
            
            # Correlation with drift
            for metric in [f"drift_{r}" for r in FLOW_RELATIONS]:
                # Drift is stored differently
                drift_vals = [trans_result["drift"].get(f"drift_{r}", 0.0) for r in FLOW_RELATIONS]
                T_vals = [trans_result["T_r"].get(r, 0.0) for r in FLOW_RELATIONS]
                if len(set(drift_vals)) > 1 and len(set(T_vals)) > 1:
                    rho, _ = spearmanr(T_vals, drift_vals)
                    print(f"  drift: rho={rho:.4f}")
            
            # Correlation with attention
            attn_vals = [trans_result["attention"].get(r, 0.0) for r in FLOW_RELATIONS]
            if len(set(attn_vals)) > 1:
                rho, _ = spearmanr(T_vals, attn_vals)
                print(f"  attention: rho={rho:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Mechanistic analysis of functional transfer")
    parser.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    parser.add_argument("--run-dir-seed42", type=Path, default=Path("runs/gnn_3layer_residual_s42"))
    parser.add_argument("--run-dir-seed1", type=Path, default=Path("runs/gnn_3layer_residual_s1"))
    parser.add_argument("--run-dir-seed2", type=Path, default=Path("runs/gnn_3layer_residual_s2"))
    parser.add_argument("--intervention-dir", type=Path, default=Path("runs/transfer_intervention"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-graphs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/mechanistic_analysis"))
    parser.add_argument("--focus-transition", type=str, help="Focus on specific transition (e.g., t3_t4)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1])
    args = parser.parse_args()
    
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    run_dirs = {}
    for seed in args.seeds:
        if seed == 42:
            run_dirs[seed] = args.run_dir_seed42
        elif seed == 1:
            run_dirs[seed] = args.run_dir_seed1
        elif seed == 2:
            run_dirs[seed] = args.run_dir_seed2
    
    if args.focus_transition:
        # Run single transition analysis
        task = int(args.focus_transition.split("_")[0][1:])
        for seed in args.seeds:
            run_dir = run_dirs.get(seed)
            if not run_dir or not run_dir.exists():
                continue
            print(f"[analysis] Focus on {args.focus_transition} for seed {seed}...")
            result = run_analysis_at_boundary(
                task, args.graphs_dir, run_dir, device, seed=seed,
                max_graphs=args.max_graphs, batch_size=args.batch_size
            )
            
            # Print detailed comparison
            print(f"\n=== {seed} - {args.focus_transition} ===")
            print(f"T_r: {result['T_r']}")
            
            # Sort relations by T_r
            sorted_rels = sorted(result['T_r'].items(), key=lambda x: x[1], reverse=True)
            print(f"Relations by T_r (high to low):")
            for r, t in sorted_rels:
                print(f"  {r}: T_r={t:.6f}")
            
            # Print all metrics per relation
            print("\nDetailed metrics per relation:")
            for relation in FLOW_RELATIONS:
                print(f"\n  {relation}:")
                print(f"    T_r: {result['T_r'].get(relation, 0):.6f}")
                geo = result['geometry'].get(relation, {})
                print(f"    geometry: intra_var={geo.get('intra_var',0):.4f}, inter_dist={geo.get('inter_dist',0):.4f}, sep_ratio={geo.get('separation_ratio',0):.4f}")
                pc = result['param_changes'].get(relation, {})
                print(f"    param_changes: norm={pc.get('param_change_norm',0):.4f}, rel_change={pc.get('relative_change',0):.4f}")
                grad = result['grad_magnitude'].get(relation, {})
                print(f"    grad_magnitude: norm={grad.get('grad_norm',0):.4f}")
                print(f"    attention: {result['attention'].get(relation, 0):.4f}")
                drift_val = result['drift'].get(f"drift_{relation}", 0)
                print(f"    drift: {drift_val:.4f}")
    else:
        # Run full analysis
        results = run_full_analysis(
            graphs_dir=args.graphs_dir,
            run_dirs=run_dirs,
            seeds=args.seeds,
            device=args.device,
            out_dir=args.out_dir,
        )
        print_correlation_summary(results)


if __name__ == "__main__":
    main()