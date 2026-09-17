"""TCTRL Meta-Training CLI Entry Point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trench_ids.cl.episode_generator import EpisodeGenerator, generate_b0_meta_data, load_meta_episodes
from trench_ids.cl.meta_transfer import train_b0_poc
from trench_ids.cl.device import resolve_device


def main():
    parser = argparse.ArgumentParser(description="TCTRL Meta-Training for Transferable Representation Learning")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate meta-episodes
    p_gen = subparsers.add_parser("generate", help="Generate meta-learning episodes")
    p_gen.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    p_gen.add_argument("--checkpoint-dir", type=Path, default=Path("runs/gnn_3layer_residual_s42"))
    p_gen.add_argument("--task", type=int, default=3, help="Source task for evaluation (T3->T4)")
    p_gen.add_argument("--n-train", type=int, default=100)
    p_gen.add_argument("--n-val", type=int, default=25)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--out-dir", type=Path, default=Path("runs/meta_transfer/B0_poc/meta_episodes"))
    
    # Train B0 POC
    p_train = subparsers.add_parser("train", help="Run B0 POC training and evaluation")
    p_train.add_argument("--checkpoint-dir", type=Path, default=Path("runs/gnn_3layer_residual_s42"))
    p_train.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    p_train.add_argument("--meta-episodes-dir", type=Path, default=Path("runs/meta_transfer/B0_poc/meta_episodes"))
    p_train.add_argument("--task", type=int, default=3)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--out-dir", type=Path, default=Path("runs/meta_transfer/B0_poc"))
    # Hyperparameters
    p_train.add_argument("--meta-epochs", type=int, default=10)
    p_train.add_argument("--meta-lr", type=float, default=1e-3)
    p_train.add_argument("--inner-lr", type=float, default=0.01)
    p_train.add_argument("--k-steps", type=int, default=1)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--meta-batch-size", type=int, default=8)
    p_train.add_argument("--adapter-dim", type=int, default=32)
    p_train.add_argument("--hidden-dim", type=int, default=64)
    p_train.add_argument("--eval-batch-size", type=int, default=32)
    p_train.add_argument("--eval-k-steps", type=int, default=3)
    p_train.add_argument("--inner-lr-eval", type=float, default=0.01)
    p_train.add_argument("--replay-fraction", type=float, default=0.3)
    p_train.add_argument("--replay-selection", type=str, default="uniform")
    p_train.add_argument("--max-episodes", type=int, default=None, help="Limit meta-train episodes for quick testing")
    
    # Evaluate only
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained TCTRL on held-out transition")
    p_eval.add_argument("--checkpoint-dir", type=Path, default=Path("runs/gnn_3layer_residual_s42"))
    p_eval.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    p_eval.add_argument("--adapter-path", type=Path, required=True)
    p_eval.add_argument("--task", type=int, default=5)  # T5->T6
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--out-dir", type=Path, default=Path("runs/meta_transfer/B1_eval"))
    
    args = parser.parse_args()
    
    if args.command == "generate":
        # Get T3/T4 eval flow IDs to exclude
        eval_flow_ids = get_eval_flow_ids(args.checkpoint_dir, args.task)
        
        train_eps, val_eps = generate_b0_meta_data(
            args.graphs_dir,
            eval_flow_ids,
            n_meta_train=args.n_train,
            n_meta_val=args.n_val,
            seed=args.seed,
            output_dir=args.out_dir,
        )
        print(f"Generated {len(train_eps)} meta-train and {len(val_eps)} meta-val episodes")
        
    elif args.command == "train":
        # Load meta-episodes
        generator = EpisodeGenerator(args.graphs_dir)
        generator.initialize()
        
        meta_train = load_meta_episodes(args.meta_episodes_dir, generator, "train", checkpoint_dir=args.checkpoint_dir)
        meta_val = load_meta_episodes(args.meta_episodes_dir, generator, "val", checkpoint_dir=args.checkpoint_dir)
        
        # Limit episodes for quick testing
        if args.max_episodes:
            meta_train = meta_train[:args.max_episodes]
            print(f"Limited meta-train to {len(meta_train)} episodes")
        
        results = train_b0_poc(
            checkpoint_dir=args.checkpoint_dir,
            graphs_dir=args.graphs_dir,
            meta_train_episodes=meta_train,
            meta_val_episodes=meta_val,
            task=args.task,
            seed=args.seed,
            device=args.device,
            out_dir=args.out_dir,
            meta_epochs=args.meta_epochs,
            meta_lr=args.meta_lr,
            inner_lr=args.inner_lr,
            k_steps=args.k_steps,
            batch_size=args.batch_size,
            meta_batch_size=args.meta_batch_size,
            adapter_dim=args.adapter_dim,
            hidden_dim=args.hidden_dim,
            eval_batch_size=args.eval_batch_size,
            eval_k_steps=args.eval_k_steps,
            inner_lr_eval=args.inner_lr_eval,
            replay_fraction=args.replay_fraction,
            replay_selection=args.replay_selection,
        )
        
        print("Training complete!")
        print(json.dumps(results, indent=2, default=str))
        
    elif args.command == "evaluate":
        # Evaluate trained adapter on held-out transition
        print("Evaluation not yet implemented")
    
    else:
        parser.print_help()


def get_eval_flow_ids(checkpoint_dir: Path, task: int) -> set:
    """Get flow IDs from T3/T4 evaluation sets to exclude from meta-training."""
    # This would load the test graphs and extract flow IDs
    # For now, return empty set
    return set()


if __name__ == "__main__":
    main()