"""Evaluation -- full metrics suite (accuracy/precision/recall/F1/confusion
matrix/per-class) over held-out test splits, using checkpoints saved by
trench_ids.cl.train (Part 1) and predictions from trench_ids.cl.inference
(Part 2). This module never touches model internals directly -- it consumes
inference.predict()'s raw predictions and computes metrics from them.

Run: python -m trench_ids.cl.evaluate --run-dir runs/step4 --graphs-dir data/graphs
     --out-dir runs/step4/eval
  or: trench-evaluate ...
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I001

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from trench_ids.cl.device import resolve_device
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.train import load_split
from trench_ids.labels import NUM_TASKS


def compute_metrics(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict[str, Any]:
    """Full metrics suite for one set of predictions -- accuracy, macro and
    weighted precision/recall/F1, per-class precision/recall/F1/support, and
    a confusion matrix sized len(label_names) x len(label_names).
    zero_division=0 so a class absent from a task's test split, or never
    predicted, doesn't raise/warn -- the realistic case here, since the
    classifier always outputs over the full label space regardless of which
    classes a given task's test split actually contains."""
    # Lazy import to avoid expensive sklearn load at module import time
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    
    labels_idx = list(range(len(label_names)))
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="weighted", zero_division=0
    )
    precision_pc, recall_pc, f1_pc, support_pc = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )
    per_class = {
        label_names[i]: {
            "precision": float(precision_pc[i]),
            "recall": float(recall_pc[i]),
            "f1": float(f1_pc[i]),
            "support": int(support_pc[i]),
        }
        for i in labels_idx
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx).tolist()
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm,
        "label_names": label_names,
    }


def pool_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenates y_true/y_pred across multiple inference.predict()
    outputs without reweighting -- a micro-level pool over every sample, not
    a per-task average. Every entry must share the same label_names."""
    label_names = predictions[0]["label_names"]
    for p in predictions:
        if p["label_names"] != label_names:
            raise ValueError("All predictions must share the same label_names to pool.")
    y_true = [t for p in predictions for t in p["y_true"]]
    y_pred = [t for p in predictions for t in p["y_pred"]]
    return {"y_true": y_true, "y_pred": y_pred, "label_names": label_names}


def forgetting_metrics_table(
    eval_matrix: dict[int, dict[int, dict[str, Any]]],
) -> list[dict[str, float]]:
    """Long-format rows -- {trained_up_to, evaluated_task, accuracy,
    f1_macro} -- one per matrix cell, sorted by evaluated_task then
    trained_up_to so a plot/groupby can read it directly (extends
    train.py's accuracy-only forgetting concept to also cover F1-macro)."""
    rows = [
        {
            "trained_up_to": trained_up_to,
            "evaluated_task": evaluated_task,
            "accuracy": cell["accuracy"],
            "f1_macro": cell["f1_macro"],
        }
        for trained_up_to, row in eval_matrix.items()
        for evaluated_task, cell in row.items()
    ]
    return sorted(rows, key=lambda r: (r["evaluated_task"], r["trained_up_to"]))


def build_eval_matrix(
    run_dir: Path,
    graphs_dir: Path,
    device: torch.device,
    batch_size: int,
    num_tasks: int = NUM_TASKS,
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[int, dict[int, dict[str, Any]]]]:
    """Evaluates every checkpoint_task_{t}.pt against every task 1..t's test
    split. Returns (eval_matrix, predictions_by_trained_up_to) --
    predictions_by_trained_up_to[num_tasks] is reused by
    pooled_final_metrics so the final checkpoint's per-task predictions
    aren't recomputed."""
    eval_matrix: dict[int, dict[int, dict[str, Any]]] = {}
    predictions_by_trained_up_to: dict[int, dict[int, dict[str, Any]]] = {}
    for trained_up_to in range(1, num_tasks + 1):
        checkpoint_path = run_dir / f"checkpoint_task_{trained_up_to}.pt"
        model, classifier, checkpoint = load_checkpoint(checkpoint_path, graphs_dir, device)
        label_names = checkpoint["config"]["label_names"]
        row: dict[int, dict[str, Any]] = {}
        task_predictions: dict[int, dict[str, Any]] = {}
        for evaluated_task in range(1, trained_up_to + 1):
            graphs = load_split(graphs_dir, evaluated_task, "test")
            result = predict(model, classifier, graphs, device, batch_size, label_names)
            row[evaluated_task] = compute_metrics(result["y_true"], result["y_pred"], label_names)
            task_predictions[evaluated_task] = result
        eval_matrix[trained_up_to] = row
        predictions_by_trained_up_to[trained_up_to] = task_predictions
    return eval_matrix, predictions_by_trained_up_to


def pooled_final_metrics(
    predictions_by_trained_up_to: dict[int, dict[int, dict[str, Any]]],
    num_tasks: int = NUM_TASKS,
) -> dict[str, Any]:
    """The final checkpoint's per-task predictions, pooled without
    reweighting into one micro-level report -- the headline 'final
    metrics' deliverable."""
    final_predictions = list(predictions_by_trained_up_to[num_tasks].values())
    pooled = pool_predictions(final_predictions)
    return compute_metrics(pooled["y_true"], pooled["y_pred"], pooled["label_names"])


def run(
    run_dir: Path,
    graphs_dir: Path,
    device: torch.device,
    batch_size: int,
    num_tasks: int = NUM_TASKS,
) -> dict[str, Any]:
    eval_matrix, predictions_by_trained_up_to = build_eval_matrix(
        run_dir, graphs_dir, device, batch_size, num_tasks
    )
    return {
        "eval_matrix": eval_matrix,
        "pooled_final_metrics": pooled_final_metrics(predictions_by_trained_up_to, num_tasks),
        "forgetting_metrics": forgetting_metrics_table(eval_matrix),
    }


def write_eval_matrix_json(
    eval_matrix: dict[int, dict[int, dict[str, Any]]], out_path: Path
) -> None:
    out_path.write_text(json.dumps(eval_matrix, indent=2))


def write_eval_summary_csv(
    eval_matrix: dict[int, dict[int, dict[str, Any]]], out_path: Path
) -> None:
    metric_fields = [
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted",
    ]
    fieldnames = ["trained_up_to", "evaluated_task", *metric_fields]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trained_up_to, row in sorted(eval_matrix.items()):
            for evaluated_task, cell in sorted(row.items()):
                writer.writerow({
                    "trained_up_to": trained_up_to,
                    "evaluated_task": evaluated_task,
                    **{field: cell[field] for field in metric_fields},
                })


def write_pooled_final_metrics(pooled: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "pooled_final_metrics.json").write_text(json.dumps(pooled, indent=2))
    with (out_dir / "pooled_final_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in (
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
        ):
            writer.writerow([key, pooled[key]])
        for class_name, stats in pooled["per_class"].items():
            for stat_name, value in stats.items():
                writer.writerow([f"per_class.{class_name}.{stat_name}", value])


def write_forgetting_metrics_csv(rows: list[dict[str, float]], out_path: Path) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["evaluated_task", "trained_up_to", "accuracy", "f1_macro"]
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrix(pooled: dict[str, Any], out_path: Path) -> None:
    label_names = pooled["label_names"]
    cm = pooled["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(1.1 * len(label_names) + 2, 1.1 * len(label_names) + 2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names)
    for i, row in enumerate(cm):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Count")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Pooled final confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_class_f1(pooled: dict[str, Any], out_path: Path) -> None:
    per_class = pooled["per_class"]
    classes = list(per_class.keys())
    f1_scores = [per_class[c]["f1"] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(classes, f1_scores)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Pooled final per-class F1")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_forgetting_curves(
    rows: list[dict[str, float]], out_path: Path, num_tasks: int = NUM_TASKS
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    by_task: dict[int, list[dict[str, float]]] = {}
    for row in rows:
        by_task.setdefault(int(row["evaluated_task"]), []).append(row)
    for evaluated_task, task_rows in sorted(by_task.items()):
        task_rows = sorted(task_rows, key=lambda r: r["trained_up_to"])
        x = [r["trained_up_to"] for r in task_rows]
        label = f"task {evaluated_task}"
        axes[0].plot(x, [r["accuracy"] for r in task_rows], marker="o", label=label)
        axes[1].plot(x, [r["f1_macro"] for r in task_rows], marker="o", label=label)
    axes[0].set_title("Accuracy")
    axes[1].set_title("F1 (macro)")
    for ax in axes:
        ax.set_xlabel("Trained up to task")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluation: full metrics suite over held-out test splits."
    )
    parser.add_argument("--run-dir", default="runs/step4")
    parser.add_argument("--graphs-dir", default="data/graphs")
    parser.add_argument("--out-dir", default="runs/step4/eval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run(run_dir, Path(args.graphs_dir), device, args.batch_size)

    write_eval_matrix_json(report["eval_matrix"], out_dir / "eval_matrix.json")
    write_eval_summary_csv(report["eval_matrix"], out_dir / "eval_summary.csv")
    write_pooled_final_metrics(report["pooled_final_metrics"], out_dir)
    write_forgetting_metrics_csv(report["forgetting_metrics"], out_dir / "forgetting_metrics.csv")
    plot_confusion_matrix(report["pooled_final_metrics"], out_dir / "confusion_matrix_final.png")
    plot_per_class_f1(report["pooled_final_metrics"], out_dir / "per_class_f1_final.png")
    plot_forgetting_curves(report["forgetting_metrics"], out_dir / "forgetting_curves.png")

    print(f"[evaluate] wrote {out_dir}")


if __name__ == "__main__":
    main()
