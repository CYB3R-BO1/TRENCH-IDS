"""Transfer Predictor: CLI for Stage A - predicting functional transfer T_r from boundary features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trench_ids.cl.transfer_predictor.data_collection import build_dataset
from trench_ids.cl.transfer_predictor.evaluate import (
    aggregate_fold_results,
    compare_models,
    print_summary,
    save_results,
)
from trench_ids.cl.transfer_predictor.loto_validation import LOTOConfig, run_loto_cv
from trench_ids.cl.transfer_predictor.models import get_all_model_names


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """Build the transfer predictor dataset from intervention results."""
    result = build_dataset(
        intervention_dir=args.intervention_dir,
        graphs_dir=args.graphs_dir,
        device=args.device,
        out_dir=args.out_dir,
    )
    print(f"[done] Built dataset with {len(result['dataset'])} records")
    return 0


def cmd_loto(args: argparse.Namespace) -> int:
    """Run Leave-One-Transition-Out cross-validation."""
    df = pd.read_csv(args.dataset)
    print(f"[loto] Loaded {len(df)} records from {args.dataset}")
    
    # Select models
    if args.models:
        models = args.models
    else:
        models = ["constant", "ridge", "rf", "gb", "mlp_small"]
    
    config = LOTOConfig(
        feature_set=args.feature_set,
        random_state=args.random_state,
    )
    
    print(f"[loto] Running models: {models}")
    print(f"[loto] Feature set: {args.feature_set}")
    
    results = run_loto_cv(df, models, config)
    
    # Aggregate and save
    aggregated = {}
    for model, res in results.items():
        if "error" not in res:
            aggregated[model] = aggregate_fold_results(res["folds"])
        else:
            aggregated[model] = res
    
    # Compare against constant baseline
    comparison = compare_models(aggregated, "constant")
    aggregated["_comparison_vs_constant"] = comparison
    
    # Print summary
    print_summary(aggregated)
    
    # Save
    out_path = args.out_dir / "loto_results.json"
    save_results(aggregated, out_path)
    print(f"\n[loto] Results saved to {out_path}")
    
    # Also save predictions CSV
    all_preds = []
    for model, res in results.items():
        if "error" in res:
            continue
        for fold_res in res["folds"]:
            if "predictions" in fold_res:
                preds = fold_res["predictions"]
                for i in range(len(preds["y_true"])):
                    all_preds.append({
                        "model": model,
                        "fold": fold_res["fold"],
                        "held_out": fold_res["held_out"],
                        "seed": preds["seeds"][i],
                        "transition": preds["transitions"][i],
                        "relation": preds["relations"][i],
                        "y_true": preds["y_true"][i],
                        "y_pred": preds["y_pred"][i],
                    })
    
    if all_preds:
        preds_df = pd.DataFrame(all_preds)
        preds_path = args.out_dir / "loto_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"[loto] Predictions saved to {preds_path}")
    
    return 0


def cmd_feature_provenance(args: argparse.Namespace) -> int:
    """Print feature provenance table."""
    from trench_ids.cl.transfer_predictor.features import print_feature_provenance_table
    print_feature_provenance_table()
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect the dataset."""
    df = pd.read_csv(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nTransitions: {sorted(df['transition'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Relations: {sorted(df['relation'].unique())}")
    print(f"\nT_r stats:")
    print(df["T_r"].describe())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Per-transition T_r stats
    print("\nT_r by transition:")
    for trans in sorted(df["transition"].unique()):
        sub = df[df["transition"] == trans]
        print(f"  {trans}: mean={sub['T_r'].mean():.6f}, std={sub['T_r'].std():.6f}, "
              f"min={sub['T_r'].min():.6f}, max={sub['T_r'].max():.6f}")
    
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transfer Predictor - Stage A: Predict functional forward transfer T_r"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # build-dataset
    p_build = subparsers.add_parser("build-dataset", help="Build dataset from intervention results")
    p_build.add_argument("--intervention-dir", type=Path, default=Path("runs/transfer_intervention"))
    p_build.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    p_build.add_argument("--device", default="auto")
    p_build.add_argument("--out-dir", type=Path, default=Path("runs/transfer_predictor"))
    p_build.set_defaults(func=cmd_build_dataset)
    
    # loto
    p_loto = subparsers.add_parser("loto", help="Run LOTO cross-validation")
    p_loto.add_argument("--dataset", type=Path, default=Path("runs/transfer_predictor/dataset.csv"))
    p_loto.add_argument("--models", nargs="+", help="Models to run (default: constant ridge rf gb mlp_small)")
    p_loto.add_argument("--feature-set", default="boundary", choices=["boundary", "all"])
    p_loto.add_argument("--random-state", type=int, default=42)
    p_loto.add_argument("--out-dir", type=Path, default=Path("runs/transfer_predictor"))
    p_loto.set_defaults(func=cmd_loto)
    
    # feature-provenance
    p_feat = subparsers.add_parser("feature-provenance", help="Print feature provenance table")
    p_feat.set_defaults(func=cmd_feature_provenance)
    
    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect dataset")
    p_inspect.add_argument("--dataset", type=Path, default=Path("runs/transfer_predictor/dataset.csv"))
    p_inspect.set_defaults(func=cmd_inspect)
    
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    exit(main())