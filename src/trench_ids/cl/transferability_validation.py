"""Transferability validation experiment: functional ablation of high/low S_r relations.

This experiment tests whether cosine-derived relation similarity scores (S_r)
from a replay-trained 3-layer residual GNN's memory bank are associated with
the functional importance of relation-specific pathways for retaining
previously learned tasks under continual learning.

Run:
    python -m trench_ids.cl.transferability_validation \
        --run-dir runs/gnn_3layer_residual_s42 \
        --graphs-dir data/graphs \
        --out-dir runs/transferability_validation/seed42 \
        --device auto
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from trench_ids.cl.device import resolve_device
from trench_ids.cl.evaluate import compute_metrics
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.transferability import aggregate_transferability_scores
from trench_ids.constants import FLOW_RELATIONS
from trench_ids.labels import attack_classes_for_task, canonical_classes
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN
from trench_ids.vocab import load_vocab, vocab_fingerprint


def load_transferability_scores(
    run_dir: Path, task: int
) -> dict[str, float]:
    """Load transferability_task_t.json and compute mean S_r per Flow relation."""
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


def get_seen_classes_up_to(task: int, label_names: list[str]) -> list[str]:
    """Return class names seen up to and including `task` (Benign + attack classes 1..task)."""
    seen = {"Benign"}
    for t in range(1, task + 1):
        seen.update(attack_classes_for_task(t))
    return [name for name in label_names if name in seen]


def get_old_classes(task: int, label_names: list[str]) -> list[str]:
    """Return class names from tasks 1..task-1 (Benign + attack classes before current task)."""
    if task <= 1:
        return ["Benign"]
    seen = {"Benign"}
    for t in range(1, task):
        seen.update(attack_classes_for_task(t))
    return [name for name in label_names if name in seen]


def evaluate_checkpoint(
    model: RelationSpecificHeteroGNN,
    classifier: torch.nn.Linear,
    graphs_dir: Path,
    device: torch.device,
    batch_size: int,
    label_names: list[str],
    eval_tasks: list[int],
    ablate_relations: set[str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Evaluate model on specified tasks with optional relation ablation."""
    results: dict[int, dict[str, Any]] = {}
    for eval_task in eval_tasks:
        graphs = torch.load(
            graphs_dir / f"task_{eval_task}_test.pt", weights_only=False
        )
        pred = predict(model, classifier, graphs, device, batch_size, label_names, ablate_relations)
        metrics = compute_metrics(pred["y_true"], pred["y_pred"], label_names)
        results[eval_task] = metrics
    return results


def filter_metrics_to_classes(
    metrics: dict[str, Any], keep_classes: list[str]
) -> dict[str, Any]:
    """Filter per-class metrics to only include specified classes."""
    filtered = dict(metrics)
    if "per_class" in metrics:
        filtered["per_class"] = {
            k: v for k, v in metrics["per_class"].items() if k in keep_classes
        }
    return filtered


def compute_old_task_aggregates(
    model: RelationSpecificHeteroGNN,
    classifier: torch.nn.Linear,
    graphs_dir: Path,
    device: torch.device,
    batch_size: int,
    label_names: list[str],
    current_task: int,
    ablate_relations: set[str] | None = None,
) -> dict[str, float]:
    """Compute aggregated metrics over old tasks 1..current_task-1."""
    old_tasks = list(range(1, current_task))
    if not old_tasks:
        return {"macro_f1": 0.0, "accuracy": 0.0}

    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for eval_task in old_tasks:
        graphs = torch.load(
            graphs_dir / f"task_{eval_task}_test.pt", weights_only=False
        )
        pred = predict(model, classifier, graphs, device, batch_size, label_names, ablate_relations)
        all_y_true.extend(pred["y_true"])
        all_y_pred.extend(pred["y_pred"])

    if not all_y_true:
        return {"macro_f1": 0.0, "accuracy": 0.0}

    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(all_y_true, all_y_pred)
    macro_f1 = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
    return {"macro_f1": float(macro_f1), "accuracy": float(acc)}


def run_permutation_test(
    s_r_scores: list[float], drops: list[float], n_permutations: int = 10000
) -> float:
    """Permutation test: shuffle S_r within each task and recompute correlation."""
    if len(s_r_scores) < 3:
        return 1.0
    observed_rho, _ = spearmanr(s_r_scores, drops)
    if np.isnan(observed_rho):
        return 1.0

    count = 0
    s_r_arr = np.array(s_r_scores)
    drops_arr = np.array(drops)

    for _ in range(n_permutations):
        perm_idx = np.random.permutation(len(s_r_arr))
        perm_rho, _ = spearmanr(s_r_arr[perm_idx], drops_arr)
        if not np.isnan(perm_rho) and abs(perm_rho) >= abs(observed_rho):
            count += 1

    return (count + 1) / (n_permutations + 1)


def select_random_relations(
    relations: list[str], k: int, seed: int
) -> list[str]:
    """Select k random relations deterministically."""
    rng = random.Random(seed)
    return rng.sample(relations, k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transferability validation: relation ablation experiment."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--graphs-dir", default="data/graphs", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-random-controls", type=int, default=3)
    parser.add_argument("--permute-test", type=int, default=10000)
    args = parser.parse_args()

    device = resolve_device(args.device)
    run_dir = args.run_dir
    graphs_dir = args.graphs_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    label_names = canonical_classes()

    ablation_results: dict[str, Any] = {}
    all_s_r: list[float] = []
    all_macro_f1_drops: list[float] = []
    all_accuracy_drops: list[float] = []
    all_current_task_f1_drops: list[float] = []

    task_wise_correlations: dict[str, Any] = {}

    for task in range(2, 7):
        checkpoint_path = run_dir / f"checkpoint_task_{task}.pt"
        if not checkpoint_path.exists():
            print(f"[warn] Missing checkpoint for task {task}")
            continue

        model, classifier, checkpoint = load_checkpoint(
            checkpoint_path, graphs_dir, device
        )
        cfg = checkpoint["config"]
        batch_size = args.batch_size or cfg.get("train", {}).get("batch_size", 8)

        s_r_dict = load_transferability_scores(run_dir, task)
        if not s_r_dict:
            print(f"[warn] No transferability scores for task {task}")
            continue

        ranked_relations = sorted(
            s_r_dict.keys(), key=lambda r: s_r_dict[r], reverse=True
        )
        print(f"[task {task}] S_r ranking: {[(r, s_r_dict[r]) for r in ranked_relations]}")

        print(f"[task {task}] Computing baseline on old tasks 1..{task-1}...")
        baseline_old = compute_old_task_aggregates(
            model, classifier, graphs_dir, device, batch_size, label_names, task, None
        )
        print(f"[task {task}] Baseline old tasks: macro_f1={baseline_old['macro_f1']:.4f}, acc={baseline_old['accuracy']:.4f}")

        baseline_current_metrics = evaluate_checkpoint(
            model, classifier, graphs_dir, device, batch_size, label_names, [task], None
        )
        baseline_current = baseline_current_metrics.get(task, {})
        print(f"[task {task}] Baseline current task: macro_f1={baseline_current.get('f1_macro', 0.0):.4f}, acc={baseline_current.get('accuracy', 0.0):.4f}")

        ablation_results[f"task_{task}"] = {
            "s_r": s_r_dict,
            "ranking": ranked_relations,
            "baseline": {
                "old_tasks": baseline_old,
                "current_task": {
                    "macro_f1": baseline_current.get("f1_macro", 0.0),
                    "accuracy": baseline_current.get("accuracy", 0.0),
                },
            },
            "ablations": {},
        }

        for rel in ranked_relations:
            print(f"  [task {task}] Ablating {rel}...")
            ablated_old = compute_old_task_aggregates(
                model, classifier, graphs_dir, device, batch_size, label_names, task, {rel}
            )
            ablated_current_metrics = evaluate_checkpoint(
                model, classifier, graphs_dir, device, batch_size, label_names, [task], {rel}
            )
            ablated_current = ablated_current_metrics.get(task, {})
            print(f"    old_task_macro_f1_drop={baseline_old['macro_f1'] - ablated_old['macro_f1']:.4f}, current_task_macro_f1_drop={baseline_current.get('f1_macro', 0.0) - ablated_current.get('f1_macro', 0.0):.4f}")

            macro_f1_drop = baseline_old["macro_f1"] - ablated_old["macro_f1"]
            accuracy_drop = baseline_old["accuracy"] - ablated_old["accuracy"]
            current_f1_drop = (
                baseline_current.get("f1_macro", 0.0)
                - ablated_current.get("f1_macro", 0.0)
            )

            all_s_r.append(s_r_dict[rel])
            all_macro_f1_drops.append(macro_f1_drop)
            all_accuracy_drops.append(accuracy_drop)
            all_current_task_f1_drops.append(current_f1_drop)

            ablation_results[f"task_{task}"]["ablations"][rel] = {
                "s_r": s_r_dict[rel],
                "old_task_macro_f1_drop": macro_f1_drop,
                "old_task_accuracy_drop": accuracy_drop,
                "current_task_macro_f1_drop": current_f1_drop,
                "old_task_ablated": ablated_old,
                "current_task_ablated": {
                    "macro_f1": ablated_current.get("f1_macro", 0.0),
                    "accuracy": ablated_current.get("accuracy", 0.0),
                },
            }

        task_s_r = [s_r_dict[rel] for rel in ranked_relations]
        task_drops = [
            ablation_results[f"task_{task}"]["ablations"][rel][
                "old_task_macro_f1_drop"
            ]
            for rel in ranked_relations
        ]
        if len(task_s_r) >= 3:
            rho, p = spearmanr(task_s_r, task_drops)
            perm_p = run_permutation_test(
                task_s_r, task_drops, n_permutations=args.permute_test
            )
            task_wise_correlations[f"task_{task}"] = {
                "spearman_rho": float(rho) if not np.isnan(rho) else None,
                "p_value": float(p) if not np.isnan(p) else None,
                "permutation_p": perm_p,
                "n": len(task_s_r),
            }

    if len(all_s_r) >= 3:
        pooled_rho, pooled_p = spearmanr(all_s_r, all_macro_f1_drops)
        pooled_perm_p = run_permutation_test(
            all_s_r, all_macro_f1_drops, n_permutations=args.permute_test
        )
        rank_correlation = {
            "pooled": {
                "spearman_rho": float(pooled_rho)
                if not np.isnan(pooled_rho)
                else None,
                "p_value": float(pooled_p) if not np.isnan(pooled_p) else None,
                "permutation_p": pooled_perm_p,
                "n": len(all_s_r),
            },
            "by_metric": {
                "old_task_macro_f1": {
                    "spearman_rho": float(pooled_rho)
                    if not np.isnan(pooled_rho)
                    else None,
                    "permutation_p": pooled_perm_p,
                },
                "old_task_accuracy": {
                    "spearman_rho": float(
                        spearmanr(all_s_r, all_accuracy_drops)[0]
                    )
                    if not np.isnan(spearmanr(all_s_r, all_accuracy_drops)[0])
                    else None,
                },
                "current_task_macro_f1": {
                    "spearman_rho": float(
                        spearmanr(all_s_r, all_current_task_f1_drops)[0]
                    )
                    if not np.isnan(spearmanr(all_s_r, all_current_task_f1_drops)[0])
                    else None,
                },
            },
            "task_wise": task_wise_correlations,
        }
    else:
        rank_correlation = {"pooled": {"n": len(all_s_r)}}

    group_ablation: dict[str, Any] = {}
    for task in range(2, 7):
        if f"task_{task}" not in ablation_results:
            continue
        s_r_dict = ablation_results[f"task_{task}"]["s_r"]
        ranked = ablation_results[f"task_{task}"]["ranking"]

        top2 = set(ranked[:2])
        bottom2 = set(ranked[-2:])

        print(f"[task {task}] Computing group ablations...")
        model, classifier, checkpoint = load_checkpoint(
            run_dir / f"checkpoint_task_{task}.pt", graphs_dir, device
        )
        cfg = checkpoint["config"]
        batch_size = args.batch_size or cfg.get("train", {}).get("batch_size", 8)

        baseline_old = compute_old_task_aggregates(
            model, classifier, graphs_dir, device, batch_size, label_names, task, None
        )

        for name, rels in [
            ("top2", top2),
            ("bottom2", bottom2),
        ]:
            print(f"  [task {task}] Group ablation: {name} ({list(rels)})")
            ablated_old = compute_old_task_aggregates(
                model, classifier, graphs_dir, device, batch_size, label_names, task, rels
            )
            group_ablation.setdefault(f"task_{task}", {})[name] = {
                "relations": list(rels),
                "old_task_macro_f1_drop": baseline_old["macro_f1"]
                - ablated_old["macro_f1"],
                "old_task_accuracy_drop": baseline_old["accuracy"]
                - ablated_old["accuracy"],
            }
            print(f"    macro_f1_drop={group_ablation[f'task_{task}'][name]['old_task_macro_f1_drop']:.4f}")

        for ctrl_idx in range(args.n_random_controls):
            seed = checkpoint.get("random_seed", 42) + task * 100 + ctrl_idx
            random_rels = set(
                select_random_relations(ranked, 2, seed)
            )
            print(f"  [task {task}] Random control {ctrl_idx}: {list(random_rels)}")
            ablated_old = compute_old_task_aggregates(
                model, classifier, graphs_dir, device, batch_size, label_names, task, random_rels
            )
            group_ablation.setdefault(f"task_{task}", {})[
                f"random2_ctrl_{ctrl_idx}"
            ] = {
                "relations": list(random_rels),
                "seed": seed,
                "old_task_macro_f1_drop": baseline_old["macro_f1"]
                - ablated_old["macro_f1"],
                "old_task_accuracy_drop": baseline_old["accuracy"]
                - ablated_old["accuracy"],
            }
            print(f"    macro_f1_drop={group_ablation[f'task_{task}'][f'random2_ctrl_{ctrl_idx}']['old_task_macro_f1_drop']:.4f}")

    (out_dir / "ablation_results.json").write_text(
        json.dumps(ablation_results, indent=2)
    )
    (out_dir / "rank_correlation.json").write_text(
        json.dumps(rank_correlation, indent=2)
    )
    (out_dir / "group_ablation.json").write_text(
        json.dumps(group_ablation, indent=2)
    )

    print(f"[done] wrote results to {out_dir}")
    if "pooled" in rank_correlation:
        r = rank_correlation["pooled"]
        print(
            f"[pooled] Spearman rho = {r.get('spearman_rho'):.4f}, "
            f"permutation p = {r.get('permutation_p'):.4f}, n = {r.get('n')}"
        )


if __name__ == "__main__":
    main()