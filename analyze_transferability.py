"""Transferability validation: does S_r(t) predict future forgetting?"""

import json
import numpy as np
from pathlib import Path

FLOW_RELATIONS = ["originates", "targeted_by", "protocol_of", "service_of", "terminated_by"]

# Task -> attack classes mapping (from labels.py)
TASK_CLASSES = {
    1: ["Scanning"],
    2: ["Reconnaissance"],
    3: ["XSS", "DDoS"],
    4: ["Password", "Infiltration"],
    5: ["DoS", "Injection"],
    6: ["Bot", "BruteForce"],
}

ALL_CLASSES = ["Benign", "Scanning", "Reconnaissance", "XSS", "DDoS", "Password", 
               "Infiltration", "DoS", "Injection", "Bot", "BruteForce"]

def load_forgetting_matrix(run_dir):
    with open(Path(run_dir) / "forgetting_matrix.json") as f:
        return {int(k): {int(kk): vv for kk, vv in v.items()} for k, v in json.load(f).items()}

def load_transferability(run_dir):
    transferability = {}
    for task in range(1, 7):
        with open(Path(run_dir) / f"transferability_task_{task}.json") as f:
            transferability[task] = json.load(f)
    return transferability

def compute_s_r(transferability, task):
    """Compute S_r for a task: mean cosine similarity across all new/old class pairs per relation."""
    report = transferability[task]
    sums = {r: 0.0 for r in FLOW_RELATIONS}
    counts = {r: 0 for r in FLOW_RELATIONS}
    for new_class, old_classes in report.items():
        for old_class, relations in old_classes.items():
            for r, cos in relations.items():
                if r in sums:
                    sums[r] += cos
                    counts[r] += 1
    return {r: (sums[r] / counts[r] if counts[r] > 0 else 0.0) for r in FLOW_RELATIONS}

def compute_forgetting_per_task(forgetting_matrix):
    """Compute per-task forgetting: peak_acc - final_acc."""
    final_task = max(forgetting_matrix.keys())
    task_forgetting = {}
    for t in range(1, final_task):
        peak = max(forgetting_matrix[tt][t] for tt in range(t, final_task))
        final = forgetting_matrix[final_task][t]
        task_forgetting[t] = peak - final
    return task_forgetting

def analyze_run(run_dir):
    print(f"\n=== {run_dir} ===")
    fm = load_forgetting_matrix(run_dir)
    tf = load_transferability(run_dir)
    
    task_forgetting = compute_forgetting_per_task(fm)
    print(f"Per-task forgetting: {task_forgetting}")
    print(f"Average forgetting: {np.mean(list(task_forgetting.values())):.4f}")
    
    # For each task, compute S_r and see if it correlates with future forgetting
    # Future forgetting = average forgetting on tasks t+1..6
    for task in range(1, 6):  # No transferability for task 6 (no future tasks)
        s_r = compute_s_r(tf, task)
        print(f"\nTask {task} S_r: {s_r}")
        
        # Future forgetting: average forgetting on subsequent tasks that have forgetting values
        future_tasks = [t for t in range(task + 1, 7) if t in task_forgetting]
        if future_tasks:
            future_forgetting = np.mean([task_forgetting[t] for t in future_tasks])
            print(f"  Future forgetting (tasks {future_tasks}): {future_forgetting:.4f}")
        
    # Cross-task correlation: for each relation, correlate S_r(t) with forgetting on task t+1
    print("\n--- Cross-task correlation (S_r(t) vs forgetting on task t+1) ---")
    for r in FLOW_RELATIONS:
        s_r_vals = []
        forget_vals = []
        for task in range(1, 6):
            s_r = compute_s_r(tf, task)
            s_r_vals.append(s_r[r])
            # Forgetting on the next task (or average of remaining)
            if task + 1 in task_forgetting:
                forget_vals.append(task_forgetting[task + 1])
            else:
                forget_vals.append(0)
        if len(s_r_vals) > 1:
            corr = np.corrcoef(s_r_vals, forget_vals)[0, 1]
            print(f"  {r}: S_r={s_r_vals}, forgetting={forget_vals}, r={corr:.3f}")

# Run analysis on all three runs
for run_dir in [
    "runs/gnn_3layer_residual_s42",
    "runs/gnn_2layer_residual_s42",
    "runs/flathost_replay_s42",
]:
    analyze_run(run_dir)

print("\n=== SUMMARY ===")
print("Architecture ablation (seed 42, replay):")
print("  3-layer GNN + residual:  forgetting=0.089, accuracy=0.896")
print("  2-layer GNN + residual:  forgetting=0.119, accuracy=0.854")
print("  Flow-only 2-layer GNN:   forgetting=0.116, accuracy=0.856")
print("  Flat+Host:               forgetting=0.091, accuracy=0.887")
print("  Original 1-layer GNN:    forgetting=0.202, accuracy=0.764")