"""Evaluation metrics and reporting for transfer predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute all evaluation metrics."""
    metrics = {}
    
    # Spearman correlation
    if len(y_true) > 1:
        rho, p = spearmanr(y_true, y_pred)
        metrics["spearman_rho"] = float(rho) if not np.isnan(rho) else None
        metrics["spearman_p"] = float(p) if not np.isnan(p) else None
    
    # Pearson correlation
    if len(y_true) > 1:
        r, p = pearsonr(y_true, y_pred)
        metrics["pearson_r"] = float(r) if not np.isnan(r) else None
        metrics["pearson_p"] = float(p) if not np.isnan(p) else None
    
    # MAE
    metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    
    # RMSE
    metrics["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # Top-1 accuracy (identifies best relation)
    if len(y_true) > 0:
        best_true_idx = np.argmax(y_true)
        best_pred_idx = np.argmax(y_pred)
        metrics["top1_accuracy"] = float(best_true_idx == best_pred_idx)
    
    # Top-k accuracy
    if len(y_true) >= 3:
        k = min(3, len(y_true))
        top_k_true = np.argsort(y_true)[-k:]
        top_k_pred = np.argsort(y_pred)[-k:]
        metrics[f"top{k}_accuracy"] = float(len(set(top_k_true) & set(top_k_pred)) / k)
    
    # Sign accuracy (correct sign of T_r)
    if len(y_true) > 0:
        sign_true = np.sign(y_true)
        sign_pred = np.sign(y_pred)
        metrics["sign_accuracy"] = float(np.mean(sign_true == sign_pred))
    
    return metrics


def compute_metrics_per_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    group_names: list[str] = None,
) -> dict[str, dict[str, float]]:
    """Compute metrics per group (e.g., per transition or per seed)."""
    if group_names is None:
        group_names = sorted(set(groups))
    
    results = {}
    for g in group_names:
        mask = groups == g
        if mask.sum() > 0:
            results[str(g)] = compute_metrics(y_true[mask], y_pred[mask])
        else:
            results[str(g)] = {}
    return results


def aggregate_fold_results(
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate results across LOTO folds."""
    valid = [r for r in fold_results if "error" not in r]
    
    if not valid:
        return {"error": "No valid folds"}
    
    agg = {
        "n_folds": len(valid),
        "folds": fold_results,
    }
    
    # Overall metrics (pool all test predictions)
    all_y_true = []
    all_y_pred = []
    all_transitions = []
    all_seeds = []
    all_relations = []
    
    for r in valid:
        all_y_true.extend(r["predictions"]["y_true"])
        all_y_pred.extend(r["predictions"]["y_pred"])
        all_transitions.extend(r["predictions"]["transitions"])
        all_seeds.extend(r["predictions"]["seeds"])
        all_relations.extend(r["predictions"]["relations"])
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_transitions = np.array(all_transitions)
    all_seeds = np.array(all_seeds)
    all_relations = np.array(all_relations)
    
    agg["overall"] = compute_metrics(all_y_true, all_y_pred)
    agg["per_transition"] = compute_metrics_per_group(
        all_y_true, all_y_pred, all_transitions
    )
    agg["per_seed"] = compute_metrics_per_group(
        all_y_true, all_y_pred, all_seeds
    )
    agg["per_relation"] = compute_metrics_per_group(
        all_y_true, all_y_pred, all_relations
    )
    
    # Mean/std of per-fold metrics
    for metric, fold_key in [
        ("spearman_rho", "spearman_rho_overall"),
        ("mae", "mae_overall"),
        ("rmse", "rmse_overall"),
        ("top1_accuracy", "top1_acc_overall"),
    ]:
        values = [r.get(fold_key) for r in valid if r.get(fold_key) is not None]
        if values:
            agg[f"fold_{metric}_mean"] = float(np.mean(values))
            agg[f"fold_{metric}_std"] = float(np.std(values))
    
    return agg


def compare_models(
    results: dict[str, Any],
    baseline_model: str = "constant",
) -> dict[str, Any]:
    """Compare models against a baseline."""
    if baseline_model not in results:
        return {"error": f"Baseline model {baseline_model} not in results"}
    
    baseline = results[baseline_model]
    if "error" in baseline:
        return {"error": "Baseline model failed"}
    
    comparison = {}
    
    for model_name, res in results.items():
        if model_name == baseline_model or "error" in res:
            continue
        
        comparison[model_name] = {
            "spearman_improvement": res.get("fold_spearman_rho_mean", 0) - baseline.get("fold_spearman_rho_mean", 0),
            "mae_improvement": baseline.get("fold_mae_mean", 0) - res.get("fold_mae_mean", 0),  # lower MAE is better
            "rmse_improvement": baseline.get("fold_rmse_mean", 0) - res.get("fold_rmse_mean", 0),
            "top1_improvement": res.get("fold_top1_accuracy_mean", 0) - baseline.get("fold_top1_accuracy_mean", 0),
        }
    
    return comparison


def print_summary(results: dict[str, Any]):
    """Print a formatted summary of results."""
    print("\n" + "=" * 80)
    print("TRANSFER PREDICTOR - LOTO VALIDATION SUMMARY")
    print("=" * 80)
    
    # Overall comparison table
    print(f"\n{'Model':<20} {'Spearman rho':>12} {'MAE':>12} {'RMSE':>12} {'Top-1':>8}")
    print("-" * 68)
    
    for model_name, res in results.items():
        if "error" in res:
            print(f"{model_name:<20} {'ERROR':>12}")
            continue
        
        rho = res.get("fold_spearman_rho_mean", None)
        mae = res.get("fold_mae_mean", None)
        rmse = res.get("fold_rmse_mean", None)
        top1 = res.get("fold_top1_accuracy_mean", None)
        
        rho_str = f"{rho:.4f} ± {res.get('fold_spearman_rho_std', 0):.4f}" if rho is not None else "N/A"
        mae_str = f"{mae:.6f} ± {res.get('fold_mae_std', 0):.6f}" if mae is not None else "N/A"
        rmse_str = f"{rmse:.6f} ± {res.get('fold_rmse_std', 0):.6f}" if rmse is not None else "N/A"
        top1_str = f"{top1:.2f}" if top1 is not None else "N/A"
        
        print(f"{model_name:<20} {rho_str:>12} {mae_str:>12} {rmse_str:>12} {top1_str:>8}")
    
    # Per-seed breakdown
    print("\nPer-seed breakdown:")
    for model_name, res in results.items():
        if "error" in res or "per_seed" not in res:
            continue
        print(f"\n  {model_name}:")
        for seed, metrics in res["per_seed"].items():
            rho = metrics.get("mean_spearman_rho")
            mae = metrics.get("mean_mae")
            top1 = metrics.get("mean_top1_acc")
            rho_str = f"{rho:.4f}" if rho is not None else "N/A"
            mae_str = f"{mae:.6f}" if mae is not None else "N/A"
            top1_str = f"{top1:.2f}" if top1 is not None else "N/A"
            print(f"    {seed}: rho = {rho_str}, MAE = {mae_str}, Top-1 = {top1_str}")
    
    # Per-transition breakdown (for best model)
    best_model = max(
        (k for k, v in results.items() if "error" not in v),
        key=lambda k: results[k].get("fold_spearman_rho_mean", -1),
        default=None
    )
    
    if best_model and "per_transition" in results[best_model]:
        print(f"\nPer-transition (best model: {best_model}):")
        for trans, metrics in results[best_model]["per_transition"].items():
            if metrics:
                rho = metrics.get("spearman_rho")
                mae = metrics.get("mae")
                top1 = metrics.get("top1_accuracy")
                rho_str = f"{rho:.4f}" if rho is not None else "N/A"
                mae_str = f"{mae:.6f}" if mae is not None else "N/A"
                top1_str = f"{top1:.2f}" if top1 is not None else "N/A"
                print(f"    {trans}: rho = {rho_str}, MAE = {mae_str}, Top-1 = {top1_str}")
    
    print("=" * 80)


def save_results(
    results: dict[str, Any],
    out_path: Path,
):
    """Save results to JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    converted = convert(results)
    out_path.write_text(json.dumps(converted, indent=2))


def load_results(in_path: Path) -> dict[str, Any]:
    """Load results from JSON file."""
    return json.loads(in_path.read_text())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and print transfer predictor results")
    parser.add_argument("--results", type=Path, default=Path("runs/transfer_predictor/loto_results.json"))
    args = parser.parse_args()
    
    results = load_results(args.results)
    print_summary(results)