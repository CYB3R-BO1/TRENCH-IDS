#!/usr/bin/env python3
"""Orchestration script for the No-Cap experiment.

Runs the full pipeline (Step 1 -> Step 2 -> Training) on the no-cap benchmark
with fixed absolute replay memory (200 graphs/task). All outputs isolated to
data/no-cap/ and runs/no-cap/.

Usage:
  python scripts/run_no_cap.py --phase preprocess   # Step 1 only
  python scripts/run_no_cap.py --phase graphs       # Step 2 only
  python scripts/run_no_cap.py --phase train        # All training phases
  python scripts/run_no_cap.py                      # Full pipeline

Phases:
  1. Fine-tuning (no replay/EWC) - seeds 42, 1, 2
  2. Replay (200 graphs/task)    - seeds 42, 1, 2
  3. EWC (lambda=1)              - seeds 42, 1, 2
  4. TRD (transfer, lambda_d=0.25) - seeds 42, 1, 2
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import yaml


def load_no_cap_config() -> dict:
    """Load no-cap config to get paths."""
    graph_cfg = yaml.safe_load(Path("configs/no-cap/graph.yaml").read_text())
    train_cfg = yaml.safe_load(Path("configs/no-cap/train.yaml").read_text())
    return {
        "graphs_dir": graph_cfg["paths"]["out_dir"],
        "out_dir_base": train_cfg["paths"]["out_dir"].replace("/step4", ""),
    }


def run_cmd(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"✓ SUCCESS: {description} ({elapsed:.1f}s)")
        return True
    else:
        print(f"✗ FAILED: {description} ({elapsed:.1f}s)")
        return False


def phase_preprocess() -> bool:
    """Step 1: No-Cap preprocessing."""
    return run_cmd(
        [
            sys.executable,
            "-m",
            "trench_ids.preprocess_no_cap",
            "--config",
            "configs/no-cap/preprocess.yaml",
        ],
        "Step 1 No-Cap Preprocessing",
    )


def phase_graphs() -> bool:
    """Step 2: Graph construction."""
    return run_cmd(
        [
            sys.executable,
            "-m",
            "trench_ids.graphs",
            "--config",
            "configs/no-cap/graph.yaml",
        ],
        "Step 2 No-Cap Graph Construction",
    )


def train_run(
    name: str,
    seed: int,
    model_overrides: list[str],
    replay: bool = False,
    ewc: bool = False,
    distill: bool = False,
    paths: dict = None,
) -> bool:
    """Run a single training configuration."""
    if paths is None:
        paths = load_no_cap_config()
    
    graphs_dir = paths["graphs_dir"]
    out_dir_base = paths["out_dir_base"]
    
    cmd = [
        sys.executable,
        "-m",
        "trench_ids.cl.train",
        f"model.num_layers=3",
        f"model.use_residual=true",
        f"train.seed={seed}",
        f"paths.graphs_dir={graphs_dir}",
        f"paths.out_dir={out_dir_base}/{name}_s{seed}",
    ]

    # Model overrides (e.g., num_layers, fusion)
    cmd.extend(model_overrides)

    # Replay
    if replay:
        cmd.extend([
            "replay.enabled=true",
            "replay.buffer_size_per_task=200",
            "replay.replay_fraction=0.3",
            "replay.selection=uniform",
        ])
    else:
        cmd.append("replay.enabled=false")

    # EWC
    if ewc:
        cmd.extend([
            "ewc.lambda_r=1",
            "ewc.lambda_s=1",
            "ewc.lambda_u=1",
            "ewc.disable_learned_weighting=false",
        ])
    else:
        cmd.extend([
            "ewc.lambda_r=0",
            "ewc.lambda_s=0",
            "ewc.lambda_u=0",
        ])

    # Distillation (TRD)
    if distill:
        cmd.extend([
            "distill.enabled=true",
            "distill.lambda_d=0.25",
            "distill.weighting=transfer",
            "distill.temperature=0.3",
            "distill.objective=both",
        ])
    else:
        cmd.append("distill.enabled=false")

    return run_cmd(cmd, f"Train {name} seed={seed}")


def phase_train() -> bool:
    """Run all training phases."""
    paths = load_no_cap_config()
    seeds = [42, 1, 2]
    success = True

    # Phase 1: Fine-tuning (baseline)
    print("\n" + "#"*60)
    print("# PHASE 1: FINE-TUNING BASELINE (no replay/EWC/TRD)")
    print("#"*60)
    for seed in seeds:
        if not train_run("gnn_finetune", seed, [], paths=paths):
            success = False

    # Phase 2: Replay (fixed 200 graphs/task - PRIMARY)
    print("\n" + "#"*60)
    print("# PHASE 2: REPLAY (200 graphs/task - FIXED ABSOLUTE MEMORY)")
    print("#"*60)
    for seed in seeds:
        if not train_run("gnn_replay", seed, [], replay=True, paths=paths):
            success = False

    # Phase 3: EWC (lambda=1)
    print("\n" + "#"*60)
    print("# PHASE 3: EWC (lambda=1)")
    print("#"*60)
    for seed in seeds:
        if not train_run("gnn_ewc", seed, [], ewc=True, paths=paths):
            success = False

    # Phase 4: TRD (transfer, lambda_d=0.25)
    print("\n" + "#"*60)
    print("# PHASE 4: TRD (transfer, lambda_d=0.25)")
    print("#"*60)
    for seed in seeds:
        if not train_run("gnn_trd_transfer", seed, [], distill=True, paths=paths):
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="Run No-Cap experiment pipeline.")
    parser.add_argument(
        "--phase",
        choices=["preprocess", "graphs", "train", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    args = parser.parse_args()

    phases = {
        "preprocess": phase_preprocess,
        "graphs": phase_graphs,
        "train": phase_train,
    }

    if args.phase == "all":
        success = True
        for name, func in [("preprocess", phase_preprocess), ("graphs", phase_graphs), ("train", phase_train)]:
            if not func():
                print(f"\n✗ Phase '{name}' failed. Stopping.")
                success = False
                break
        if success:
            print("\n" + "="*60)
            print("ALL PHASES COMPLETED SUCCESSFULLY")
            print("="*60)
        sys.exit(0 if success else 1)
    else:
        success = phases[args.phase]()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()