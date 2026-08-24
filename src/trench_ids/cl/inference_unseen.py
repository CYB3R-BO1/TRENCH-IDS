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

from trench_ids.cl.device import resolve_device
from trench_ids.cl.evaluate import compute_metrics, pool_predictions
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
    # Dataset B's whole pipeline (compute_class_relation_means, then
    # similarity_report/estimate_transferability) depends on per-relation
    # Flow embeddings, which only RelationSpecificHeteroGNN produces --
    # FlatFlowEncoder.forward always returns relations={"flow": {}} by
    # design (model/flat.py), so pointing this at a train_flat.py checkpoint
    # would otherwise degrade silently: compute_class_relation_means's inner
    # loop runs zero times, dataset_b_means and every downstream similarity
    # file come out empty, and nothing raises.
    model_type = checkpoint["config"].get("model_type", "rhgnn")
    if model_type != "rhgnn":
        raise ValueError(
            f"inference_unseen needs a RelationSpecificHeteroGNN checkpoint (per-relation "
            f"embeddings for Dataset B's similarity report), but {checkpoint_path} has "
            f"model_type={model_type!r}. Point --checkpoint at a GNN run instead."
        )
    label_names = checkpoint["config"]["label_names"]

    dataset_a: list[dict[str, Any]] = []
    dataset_a_results: list[dict[str, Any]] = []
    dataset_b_predictions: dict[str, Any] = {}
    dataset_b_confidence: dict[str, Any] = {}
    # Keyed on slug (not canonical_label) so it joins with dataset_b_predictions/
    # dataset_b_confidence -- see Finding #3. Using canonical_label here would
    # also risk silently merging distinct manifest entries that happen to share
    # a canonical_label (e.g. Dataset A's two "DoS" entries), even though that
    # collision can't currently occur within Dataset B specifically.
    dataset_b_means: dict[str, dict[str, torch.Tensor]] = {}
    slug_to_canonical: dict[str, str] = {}

    for slug, entry in manifest["classes"].items():
        graphs = torch.load(Path(unseen_graphs_dir) / f"{slug}.pt", weights_only=False)
        result = predict(model, classifier, graphs, device, batch_size, label_names)

        if entry["evaluation_type"] == "seen_class_unseen_samples":
            metrics = compute_metrics(result["y_true"], result["y_pred"], label_names)
            dataset_a.append({"slug": slug, **entry, "metrics": metrics})
            dataset_a_results.append(result)
        else:
            canonical_label = entry["canonical_label"]
            slug_to_canonical[slug] = canonical_label

            pred_rows = prediction_distribution(result["y_pred"], label_names)
            for row in pred_rows:
                row["unseen_class"] = slug
                row["canonical_label"] = canonical_label
            dataset_b_predictions[slug] = pred_rows

            conf = confidence_stats(result["y_prob"])
            conf["unseen_class"] = slug
            conf["canonical_label"] = canonical_label
            dataset_b_confidence[slug] = conf

            dataset_b_means[slug] = compute_class_relation_means(
                model, graphs, device, batch_size
            )

    # Pooled Dataset A metrics -- micro-averaged over every Dataset A sample,
    # alongside (not replacing) the per-class list already built above.
    dataset_a_pooled_metrics: dict[str, Any] = {}
    if dataset_a_results:
        pooled = pool_predictions(dataset_a_results)
        dataset_a_pooled_metrics = compute_metrics(
            pooled["y_true"], pooled["y_pred"], pooled["label_names"]
        )

    bank = load_memory_bank(Path(memory_bank_path), map_location=device)
    dataset_b_similarity = similarity_report(dataset_b_means, bank)  # keyed by slug
    dataset_b_similarity_out = {
        slug: {"canonical_label": slug_to_canonical[slug], "similarity": by_relation}
        for slug, by_relation in dataset_b_similarity.items()
    }

    (out_dir / "dataset_a_metrics.json").write_text(
        json.dumps({"per_class": dataset_a, "pooled": dataset_a_pooled_metrics}, indent=2)
    )
    (out_dir / "dataset_b_predictions.json").write_text(json.dumps(dataset_b_predictions, indent=2))
    (out_dir / "dataset_b_confidence.json").write_text(json.dumps(dataset_b_confidence, indent=2))
    (out_dir / "dataset_b_similarity.json").write_text(
        json.dumps(dataset_b_similarity_out, indent=2)
    )

    with open(out_dir / "dataset_b_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unseen_class", "canonical_label", "predicted_class", "percentage"])
        for slug, rows in dataset_b_predictions.items():
            for row in rows:
                writer.writerow(
                    [slug, slug_to_canonical[slug], row["predicted_class"], row["percentage"]]
                )

    with open(out_dir / "dataset_b_confidence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unseen_class",
                "canonical_label",
                "mean_max_softmax",
                "median_max_softmax",
                "std_max_softmax",
            ]
        )
        for slug, stats in dataset_b_confidence.items():
            writer.writerow(
                [
                    slug,
                    slug_to_canonical[slug],
                    stats["mean_max_softmax"],
                    stats["median_max_softmax"],
                    stats["std_max_softmax"],
                ]
            )

    with open(out_dir / "dataset_b_similarity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["unseen_class", "canonical_label", "relation", "known_class", "cosine_similarity"]
        )
        for slug, relation, known_class, cosine in similarity_rows(dataset_b_similarity):
            writer.writerow([slug, slug_to_canonical[slug], relation, known_class, cosine])

    return {
        "dataset_a": dataset_a,
        "dataset_a_pooled": dataset_a_pooled_metrics,
        "dataset_b_predictions": dataset_b_predictions,
        "dataset_b_confidence": dataset_b_confidence,
        "dataset_b_similarity": dataset_b_similarity_out,
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

    device = resolve_device(args.device)

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
