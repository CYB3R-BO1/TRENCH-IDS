"""Leave-One-Transition-Out (LOTO) validation for transfer predictor.

Implements 5-fold cross-validation where each fold holds out one task transition
(both seeds) as the test set, and trains on the other 4 transitions (both seeds).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="An input array is constant")
warnings.filterwarnings("ignore", category=UserWarning, message="An input array is constant")

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trench_ids.cl.transfer_predictor.features import build_feature_matrix


@dataclass
class LOTOConfig:
    """Configuration for LOTO validation."""
    feature_set: str = "boundary"  # "boundary", "all", or list of feature names
    random_state: int = 42


# Define the 5 LOTO folds
# Each fold holds out one transition (both seeds)
LOTO_FOLDS = [
    {
        "name": "fold_T1_T2",
        "held_out_transition": "t1_t2",
        "train_transitions": ["t2_t3", "t3_t4", "t4_t5", "t5_t6"],
    },
    {
        "name": "fold_T2_T3",
        "held_out_transition": "t2_t3",
        "train_transitions": ["t1_t2", "t3_t4", "t4_t5", "t5_t6"],
    },
    {
        "name": "fold_T3_T4",
        "held_out_transition": "t3_t4",
        "train_transitions": ["t1_t2", "t2_t3", "t4_t5", "t5_t6"],
    },
    {
        "name": "fold_T4_T5",
        "held_out_transition": "t4_t5",
        "train_transitions": ["t1_t2", "t2_t3", "t3_t4", "t5_t6"],
    },
    {
        "name": "fold_T5_T6",
        "held_out_transition": "t5_t6",
        "train_transitions": ["t1_t2", "t2_t3", "t3_t4", "t4_t5"],
    },
]


def get_model(model_name: str, random_state: int = 42) -> BaseEstimator:
    """Get model by name."""
    models = {
        "constant": None,  # Special case - mean predictor
        "ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0, random_state=random_state)
        ),
        "ridge_cv": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0, random_state=random_state)
        ),
        "rf": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gb": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=0.01,
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
            )
        ),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    if model_name == "constant":
        return None
    
    return models[model_name]


def predict_constant(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Constant predictor: always predict mean of training targets."""
    return np.full(X_test.shape[0], y_train.mean())


def run_loto_fold(
    df: pd.DataFrame,
    fold: dict[str, Any],
    model_name: str,
    config: LOTOConfig,
) -> dict[str, Any]:
    """Run a single LOTO fold."""
    held_out = fold["held_out_transition"]
    train_transitions = fold["train_transitions"]
    
    # Split data
    train_mask = df["transition"].isin(train_transitions)
    test_mask = df["transition"] == held_out
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    if len(df_train) == 0 or len(df_test) == 0:
        return {
            "fold": fold["name"],
            "held_out": held_out,
            "error": "Empty train or test set",
        }
    
    # Build feature matrices
    X_train, y_train, feature_names = build_feature_matrix(df_train, config.feature_set)
    X_test, y_test, _ = build_feature_matrix(df_test, config.feature_set)
    
    # Train and predict
    if model_name == "constant":
        y_pred = predict_constant(X_train, y_train, X_test)
    else:
        model = get_model(model_name, config.random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Compute metrics
    from scipy.stats import spearmanr, pearsonr
    
    results = {
        "fold": fold["name"],
        "held_out": held_out,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "feature_names": feature_names,
    }
    
    # Per-seed results
    for seed in [42, 1]:
        seed_mask = df_test["seed"] == seed
        if seed_mask.sum() > 0:
            y_seed = y_test[seed_mask]
            y_pred_seed = y_pred[seed_mask]
            
            # Spearman
            if len(y_seed) > 1 and np.std(y_seed) > 0 and np.std(y_pred_seed) > 0:
                rho, p = spearmanr(y_seed, y_pred_seed)
                results[f"spearman_rho_seed{seed}"] = float(rho) if not np.isnan(rho) else None
                results[f"spearman_p_seed{seed}"] = float(p) if not np.isnan(p) else None
            else:
                results[f"spearman_rho_seed{seed}"] = None
                results[f"spearman_p_seed{seed}"] = None
            
            # Pearson
            if len(y_seed) > 1 and np.std(y_seed) > 0 and np.std(y_pred_seed) > 0:
                r, p = pearsonr(y_seed, y_pred_seed)
                results[f"pearson_r_seed{seed}"] = float(r) if not np.isnan(r) else None
                results[f"pearson_p_seed{seed}"] = float(p) if not np.isnan(p) else None
            else:
                results[f"pearson_r_seed{seed}"] = None
                results[f"pearson_p_seed{seed}"] = None
            
            # MAE, RMSE
            results[f"mae_seed{seed}"] = float(np.mean(np.abs(y_seed - y_pred_seed)))
            results[f"rmse_seed{seed}"] = float(np.sqrt(np.mean((y_seed - y_pred_seed) ** 2)))
            
            # Top-1 accuracy: does it correctly identify the best relation?
            # Group by transition (should be single transition per seed in test)
            best_true_idx = np.argmax(y_seed)
            best_pred_idx = np.argmax(y_pred_seed)
            results[f"top1_acc_seed{seed}"] = float(best_true_idx == best_pred_idx)
    
    # Overall (both seeds combined)
    if len(y_test) > 1:
        rho, p = spearmanr(y_test, y_pred)
        results["spearman_rho_overall"] = float(rho) if not np.isnan(rho) else None
        results["spearman_p_overall"] = float(p) if not np.isnan(p) else None
        
        r, p = pearsonr(y_test, y_pred)
        results["pearson_r_overall"] = float(r) if not np.isnan(r) else None
        results["pearson_p_overall"] = float(p) if not np.isnan(p) else None
    else:
        results["spearman_rho_overall"] = None
        results["spearman_p_overall"] = None
        results["pearson_r_overall"] = None
        results["pearson_p_overall"] = None
    
    results["mae_overall"] = float(np.mean(np.abs(y_test - y_pred)))
    results["rmse_overall"] = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    
    # Top-1 overall
    best_true_idx = np.argmax(y_test)
    best_pred_idx = np.argmax(y_pred)
    results["top1_acc_overall"] = float(best_true_idx == best_pred_idx)
    
    # Store predictions for analysis
    results["predictions"] = {
        "y_true": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "transitions": df_test["transition"].tolist(),
        "relations": df_test["relation"].tolist(),
        "seeds": df_test["seed"].tolist(),
    }
    
    return results


def run_loto_cv(
    df: pd.DataFrame,
    model_names: list[str],
    config: LOTOConfig = None,
) -> dict[str, Any]:
    """Run full LOTO cross-validation for multiple models."""
    if config is None:
        config = LOTOConfig()
    
    all_results = {}
    
    for model_name in model_names:
        print(f"[LOTO] Running model: {model_name}")
        fold_results = []
        
        for fold in LOTO_FOLDS:
            result = run_loto_fold(df, fold, model_name, config)
            fold_results.append(result)
            if "error" in result:
                print(f"  {fold['name']}: ERROR - {result['error']}")
            else:
                rho = result.get('spearman_rho_overall')
                mae = result.get('mae_overall')
                top1 = result.get('top1_acc_overall')
                rho_str = f"{rho:.4f}" if rho is not None else "N/A"
                print(f"  {fold['name']}: Spearman rho = {rho_str}, MAE = {mae:.6f}, Top-1 = {top1:.2f}")
        
        # Aggregate across folds
        valid_folds = [r for r in fold_results if "error" not in r]
        
        if valid_folds:
            def safe_mean(values):
                vals = [v for v in values if v is not None]
                return float(np.mean(vals)) if vals else None
            
            def safe_std(values):
                vals = [v for v in values if v is not None]
                return float(np.std(vals)) if len(vals) > 1 else 0.0
            
            agg = {
                "model": model_name,
                "folds": fold_results,
                "mean_spearman_rho": safe_mean([r.get("spearman_rho_overall") for r in valid_folds]),
                "std_spearman_rho": safe_std([r.get("spearman_rho_overall") for r in valid_folds]),
                "mean_mae": np.mean([r["mae_overall"] for r in valid_folds]),
                "std_mae": np.std([r["mae_overall"] for r in valid_folds]),
                "mean_rmse": np.mean([r["rmse_overall"] for r in valid_folds]),
                "mean_top1_acc": np.mean([r["top1_acc_overall"] for r in valid_folds]),
                "per_seed": {},
            }
            
            for seed in [42, 1]:
                seed_rhos = [r.get(f"spearman_rho_seed{seed}") for r in valid_folds if r.get(f"spearman_rho_seed{seed}") is not None]
                seed_maes = [r.get(f"mae_seed{seed}") for r in valid_folds if r.get(f"mae_seed{seed}") is not None]
                seed_top1 = [r.get(f"top1_acc_seed{seed}") for r in valid_folds if r.get(f"top1_acc_seed{seed}") is not None]
                
                if seed_rhos:
                    agg["per_seed"][f"seed{seed}"] = {
                        "mean_spearman_rho": safe_mean(seed_rhos),
                        "mean_mae": float(np.mean(seed_maes)) if seed_maes else None,
                        "mean_top1_acc": float(np.mean(seed_top1)) if seed_top1 else None,
                    }
        else:
            agg = {"model": model_name, "folds": fold_results, "error": "All folds failed"}
        
        all_results[model_name] = agg
    
    return all_results


def run_permutation_test(
    df: pd.DataFrame,
    model_name: str,
    config: LOTOConfig,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run permutation test for significance of Spearman correlation."""
    rng = np.random.RandomState(random_state)
    
    # Get observed overall Spearman
    fold_results = []
    for fold in LOTO_FOLDS:
        result = run_loto_fold(df, fold, model_name, config)
        if "error" not in result:
            fold_results.append(result)
    
    if not fold_results:
        return {"error": "No valid folds"}
    
    # Pool all test predictions
    all_y_true = []
    all_y_pred = []
    for r in fold_results:
        all_y_true.extend(r["predictions"]["y_true"])
        all_y_pred.extend(r["predictions"]["y_pred"])
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    observed_rho, _ = spearmanr(all_y_true, all_y_pred)
    
    # Permutation test
    null_rhos = []
    for _ in range(n_permutations):
        permuted_pred = rng.permutation(all_y_pred)
        rho, _ = spearmanr(all_y_true, permuted_pred)
        if not np.isnan(rho):
            null_rhos.append(rho)
    
    null_rhos = np.array(null_rhos)
    p_value = float((np.abs(null_rhos) >= np.abs(observed_rho)).mean())
    
    return {
        "observed_rho": float(observed_rho),
        "p_value": p_value,
        "null_mean": float(null_rhos.mean()),
        "null_std": float(null_rhos.std()),
        "n_permutations": n_permutations,
    }


if __name__ == "__main__":
    import argparse
    from trench_ids.cl.transfer_predictor.data_collection import build_dataset
    
    parser = argparse.ArgumentParser(description="Run LOTO validation")
    parser.add_argument("--dataset", type=str, default="runs/transfer_predictor/dataset.csv")
    parser.add_argument("--models", nargs="+", default=["constant", "ridge", "rf", "gb"])
    parser.add_argument("--feature-set", default="boundary", choices=["boundary", "all"])
    parser.add_argument("--out-dir", type=str, default="runs/transfer_predictor")
    args = parser.parse_args()
    
    df = pd.read_csv(args.dataset)
    config = LOTOConfig(feature_set=args.feature_set)
    
    results = run_loto_cv(df, args.models, config)
    
    import json
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loto_results.json").write_text(json.dumps(results, indent=2))
    
    print("\n[LOTO] Summary:")
    for model, res in results.items():
        if "error" not in res:
            print(f"  {model}: ρ = {res['mean_spearman_rho']:.4f} ± {res['std_spearman_rho']:.4f}, "
                  f"MAE = {res['mean_mae']:.6f} ± {res['std_mae']:.6f}, "
                  f"Top-1 = {res['mean_top1_acc']:.2f}")