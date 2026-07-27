"""Step 10 (third bullet) -- inference and behavioral analysis on
unseen-attack data (trench_ids.unseen_graphs's output).

Dataset A (evaluation_type="seen_class_unseen_samples") is scored with
ordinary classification metrics (trench_ids.cl.evaluate.compute_metrics) --
a ground-truth label exists in the classifier's output space.

Dataset B (evaluation_type="unseen_class") does NOT evaluate classification
accuracy -- there is no output neuron for an unseen class, so accuracy is
undefined. Instead it characterizes representation behavior via:

  1. Prediction distribution -- full percentage breakdown of which known
     class each sample's argmax prediction lands on.
  2. Confidence statistics -- mean/median/std of max-softmax probability.
  3. Relation-specific embedding similarity -- reuses
     trench_ids.cl.transferability's existing per-relation cosine-
     similarity machinery, pointed at Dataset B's graphs instead of a new
     task's graphs. Same-relation-only comparison.

No dedicated OOD/novelty detector is built here -- that is out of scope
(see docs/superpowers/specs/2026-07-27-unseen-attack-inference-design.md).

Run: trench-infer-unseen --checkpoint runs/replay_seed42_newport/checkpoint_task_6.pt \
    --unseen-graphs-dir data/unseen/graphs --out-dir runs/unseen_eval
  or: python -m trench_ids.cl.inference_unseen ...
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from trench_ids.cl.evaluate import compute_metrics
from trench_ids.cl.inference import load_checkpoint, predict
from trench_ids.cl.memory_bank import load_memory_bank
from trench_ids.cl.transferability import estimate_transferability
from trench_ids.model.rhgnn import RelationSpecificHeteroGNN


def prediction_distribution(y_pred: list[int], label_names: list[str]) -> list[dict[str, Any]]:
    """Full percentage breakdown of predicted-class distribution, ranked
    most to least frequent -- not just the argmax mode, so a diffuse
    distribution reads differently from a concentrated one."""
    total = len(y_pred)
    if total == 0:
        return []
    counts = Counter(y_pred)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {
            "predicted_class": label_names[idx],
            "count": count,
            "percentage": round(100.0 * count / total, 4),
        }
        for idx, count in ranked
    ]


def confidence_stats(y_prob: list[list[float]]) -> dict[str, float]:
    """Mean/median/std of max-softmax probability. The standard deviation
    distinguishes "consistently confident" from "occasionally confident"."""
    if not y_prob:
        return {"mean_max_softmax": 0.0, "median_max_softmax": 0.0, "std_max_softmax": 0.0}
    max_probs = np.array([max(row) for row in y_prob])
    return {
        "mean_max_softmax": float(max_probs.mean()),
        "median_max_softmax": float(np.median(max_probs)),
        "std_max_softmax": float(max_probs.std()),
    }


def compute_class_relation_means(
    model: RelationSpecificHeteroGNN,
    graphs: list[HeteroData],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Mean per-relation Flow embedding across every row in `graphs`.

    Unlike trench_ids.cl.memory_bank.RelationMeanAccumulator (which masks
    per-row by known-class index via graph["flow"].y), no masking is
    needed or possible here: every graph in this list belongs to exactly
    one unseen class (trench_ids.unseen_data writes one parquet, and
    trench_ids.unseen_graphs one .pt file, per class), and Dataset B's y is
    the -1 sentinel, not a usable class index.
    """
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    count = 0
    loader = DataLoader(graphs, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            n = batch["flow"].y.shape[0]
            count += n
            for relation, embed in output.relations["flow"].items():
                contribution = embed.sum(dim=0).detach()
                if relation in sums:
                    sums[relation] = sums[relation] + contribution
                else:
                    sums[relation] = contribution.clone()
    return {relation: total / count for relation, total in sums.items()}


def similarity_report(
    unseen_means: dict[str, dict[str, torch.Tensor]],
    bank: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Re-nests transferability.estimate_transferability's
    {unseen_class: {known_class: {relation: cosine}}} output into
    {unseen_class: {relation: {known_class: cosine}}}, the shape declared
    in the design spec (grouped by relation first, since that's how a
    reader compares one relation across known classes)."""
    raw = estimate_transferability(unseen_means, bank)
    nested: dict[str, dict[str, dict[str, float]]] = {}
    for unseen_class, by_known_class in raw.items():
        nested[unseen_class] = {}
        for known_class, by_relation in by_known_class.items():
            for relation, cosine in by_relation.items():
                nested[unseen_class].setdefault(relation, {})[known_class] = cosine
    return nested


def similarity_rows(
    nested: dict[str, dict[str, dict[str, float]]],
) -> list[tuple[str, str, str, float]]:
    return [
        (unseen_class, relation, known_class, cosine)
        for unseen_class, by_relation in nested.items()
        for relation, by_known_class in by_relation.items()
        for known_class, cosine in by_known_class.items()
    ]


def run(
    checkpoint_path: Path,
    graphs_dir_for_schema: Path,
    unseen_graphs_dir: Path,
    unseen_manifest_path: Path,
    memory_bank_path: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(unseen_manifest_path).read_text())
    model, classifier, checkpoint = load_checkpoint(
        Path(checkpoint_path), Path(graphs_dir_for_schema), device
    )
    label_names = checkpoint["config"]["label_names"]

    dataset_a: list[dict[str, Any]] = []
    dataset_b_predictions: dict[str, Any] = {}
    dataset_b_confidence: dict[str, Any] = {}
    dataset_b_means: dict[str, dict[str, torch.Tensor]] = {}

    for slug, entry in manifest["classes"].items():
        graphs = torch.load(Path(unseen_graphs_dir) / f"{slug}.pt", weights_only=False)
        result = predict(model, classifier, graphs, device, batch_size, label_names)

        if entry["evaluation_type"] == "seen_class_unseen_samples":
            metrics = compute_metrics(result["y_true"], result["y_pred"], label_names)
            dataset_a.append({"slug": slug, **entry, "metrics": metrics})
        else:
            dataset_b_predictions[slug] = prediction_distribution(result["y_pred"], label_names)
            dataset_b_confidence[slug] = confidence_stats(result["y_prob"])
            dataset_b_means[entry["canonical_label"]] = compute_class_relation_means(
                model, graphs, device, batch_size
            )

    bank = load_memory_bank(Path(memory_bank_path))
    dataset_b_similarity = similarity_report(dataset_b_means, bank)

    (out_dir / "dataset_a_metrics.json").write_text(json.dumps(dataset_a, indent=2))
    (out_dir / "dataset_b_predictions.json").write_text(json.dumps(dataset_b_predictions, indent=2))
    (out_dir / "dataset_b_confidence.json").write_text(json.dumps(dataset_b_confidence, indent=2))
    (out_dir / "dataset_b_similarity.json").write_text(json.dumps(dataset_b_similarity, indent=2))

    with open(out_dir / "dataset_b_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unseen_class", "predicted_class", "percentage"])
        for slug, rows in dataset_b_predictions.items():
            for row in rows:
                writer.writerow([slug, row["predicted_class"], row["percentage"]])

    with open(out_dir / "dataset_b_confidence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["unseen_class", "mean_max_softmax", "median_max_softmax", "std_max_softmax"]
        )
        for slug, stats in dataset_b_confidence.items():
            writer.writerow(
                [
                    slug,
                    stats["mean_max_softmax"],
                    stats["median_max_softmax"],
                    stats["std_max_softmax"],
                ]
            )

    with open(out_dir / "dataset_b_similarity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unseen_class", "relation", "known_class", "cosine_similarity"])
        for row in similarity_rows(dataset_b_similarity):
            writer.writerow(row)

    return {
        "dataset_a": dataset_a,
        "dataset_b_predictions": dataset_b_predictions,
        "dataset_b_confidence": dataset_b_confidence,
        "dataset_b_similarity": dataset_b_similarity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 10 -- unseen-attack inference.")
    parser.add_argument(
        "--checkpoint", default="runs/replay_seed42_newport/checkpoint_task_6.pt"
    )
    parser.add_argument("--graphs-dir", default="data/graphs")
    parser.add_argument("--unseen-graphs-dir", default="data/unseen/graphs")
    parser.add_argument("--unseen-manifest", default="data/unseen/manifest.json")
    parser.add_argument("--memory-bank", default="runs/replay_seed42_newport/memory_bank.pt")
    parser.add_argument("--out-dir", default="runs/unseen_eval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    run(
        checkpoint_path=Path(args.checkpoint),
        graphs_dir_for_schema=Path(args.graphs_dir),
        unseen_graphs_dir=Path(args.unseen_graphs_dir),
        unseen_manifest_path=Path(args.unseen_manifest),
        memory_bank_path=Path(args.memory_bank),
        out_dir=Path(args.out_dir),
        device=device,
        batch_size=args.batch_size,
    )
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
