"""Step 10a — transferability analysis (docs/Updated_Proposed_Methodology_
Transferable_Representation_Learning.docx, Step 10, first two bullets: which
learned representations transfer across attack classes, and how relation-wise
transferability evolves across successive tasks). Consolidates the frozen
runs/step4 baseline's per-task transferability_task_*.json files -- no new
training run is required, all source data already exists on disk.

The prediction/inference pipeline (Step 10's third bullet) is explicitly out
of scope here -- deferred to a separate future round.

Run: python -m trench_ids.cl.transferability_report
  or: trench-transferability-report
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402, I001

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from trench_ids.labels import NUM_TASKS


def load_transferability_records(
    run_dir: Path, num_tasks: int = NUM_TASKS
) -> list[dict[str, Any]]:
    """Flattens transferability_task_{1..num_tasks}.json into rows. Task 1
    contributes nothing (empty bank -- {"Scanning": {}} in the real data).
    Each unordered class pair appears exactly once, in the direction
    dictated by task order (the newer class is always "new_class")."""
    records: list[dict[str, Any]] = []
    run_dir = Path(run_dir)
    for task in range(1, num_tasks + 1):
        data = json.loads((run_dir / f"transferability_task_{task}.json").read_text())
        for new_class, bank_entries in data.items():
            for bank_class, per_relation in bank_entries.items():
                for relation, cosine in per_relation.items():
                    records.append({
                        "task": task,
                        "new_class": new_class,
                        "bank_class": bank_class,
                        "relation": relation,
                        "cosine": cosine,
                    })
    return records


def relation_wise_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """{relation: {"mean", "std", "n"}} across every recorded pair,
    regardless of task. "std" is population standard deviation
    (statistics.pstdev), matching numpy's default ddof=0."""
    by_relation: dict[str, list[float]] = {}
    for record in records:
        by_relation.setdefault(record["relation"], []).append(record["cosine"])
    return {
        relation: {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }
        for relation, values in by_relation.items()
    }


def relation_wise_task_evolution(records: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    """{relation: {task: mean_cosine_that_task}} -- only tasks where that
    relation has at least one observation that task."""
    by_relation_task: dict[str, dict[int, list[float]]] = {}
    for record in records:
        by_relation_task.setdefault(record["relation"], {}).setdefault(
            record["task"], []
        ).append(record["cosine"])
    return {
        relation: {task: statistics.fmean(values) for task, values in task_map.items()}
        for relation, task_map in by_relation_task.items()
    }


def rank_relations(summary: dict[str, dict[str, float | int]]) -> list[dict[str, Any]]:
    """[{relation, mean, std, n}, ...] sorted by mean descending."""
    return sorted(
        (
            {"relation": relation, "mean": stats["mean"], "std": stats["std"], "n": stats["n"]}
            for relation, stats in summary.items()
        ),
        key=lambda row: row["mean"],
        reverse=True,
    )


def class_pair_summary(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Keyed by tuple(sorted((new_class, bank_class))) -- normalized,
    order-independent, so lookups against the raw-feature matrix (also
    symmetric) can't silently miss a pair due to argument order. Missing
    relations for a pair are simply absent from "per_relation", never
    zero-filled."""
    per_pair_relations: dict[tuple[str, str], dict[str, float]] = {}
    pair_classes: dict[tuple[str, str], tuple[str, str]] = {}
    for record in records:
        key = tuple(sorted((record["new_class"], record["bank_class"])))
        per_pair_relations.setdefault(key, {})[record["relation"]] = record["cosine"]
        pair_classes[key] = (record["new_class"], record["bank_class"])

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, per_relation in per_pair_relations.items():
        new_class, bank_class = pair_classes[key]
        result[key] = {
            "new_class": new_class,
            "bank_class": bank_class,
            "mean_across_relations": statistics.fmean(per_relation.values()),
            "per_relation": per_relation,
            "best_relation": max(per_relation, key=per_relation.get),
            "worst_relation": min(per_relation, key=per_relation.get),
        }
    return result


def rank_class_pairs(summary: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """[{class_a, class_b, mean, per_relation, best_relation, worst_relation}, ...]
    sorted by mean descending."""
    return sorted(
        (
            {
                "class_a": key[0],
                "class_b": key[1],
                "mean": value["mean_across_relations"],
                "per_relation": value["per_relation"],
                "best_relation": value["best_relation"],
                "worst_relation": value["worst_relation"],
            }
            for key, value in summary.items()
        ),
        key=lambda row: row["mean"],
        reverse=True,
    )


def compare_to_raw_feature_similarity(
    pair_summary: dict[tuple[str, str], dict[str, Any]],
    raw_matrix: pd.DataFrame,
) -> dict[str, Any]:
    """Joins every class_pair_summary pair against the raw-feature matrix
    (raw_matrix is symmetric, so a plain .loc lookup on either class order
    is safe). Pearson r and Spearman rho are reported as descriptive
    statistics, not a pass/fail metric -- only partial agreement with raw
    input feature similarity is expected, since the learned representation
    is relation-specific and shaped by continual learning."""
    matched: list[dict[str, Any]] = []
    for (class_a, class_b), value in pair_summary.items():
        if class_a not in raw_matrix.index or class_b not in raw_matrix.columns:
            continue
        matched.append({
            "class_a": class_a,
            "class_b": class_b,
            "raw_cosine": float(raw_matrix.loc[class_a, class_b]),
            "learned_mean": value["mean_across_relations"],
        })

    if len(matched) >= 2:
        raw_values = [m["raw_cosine"] for m in matched]
        learned_values = [m["learned_mean"] for m in matched]
        pearson_r = float(pearsonr(raw_values, learned_values)[0])
        spearman_r = float(spearmanr(raw_values, learned_values)[0])
    else:
        pearson_r = float("nan")
        spearman_r = float("nan")

    # "Agreement" means the two scores are close for that pair -- the exact
    # complement of top_disagreements' largest-|difference| ranking. (The
    # previous version ranked by learned score alone among positively-similar
    # pairs, which measures nothing about agreement: the ordering would be
    # identical for any raw matrix.)
    def _agreement_gap(m: dict[str, Any]) -> float:
        return abs(m["raw_cosine"] - m["learned_mean"])

    top_agreements = [
        {**m, "gap": _agreement_gap(m)}
        for m in sorted(matched, key=_agreement_gap)[:5]
    ]

    def _tag(m: dict[str, Any]) -> str:
        if m["raw_cosine"] > m["learned_mean"]:
            return "high_raw_low_learned"
        return "low_raw_high_learned"

    top_disagreements = [
        {**m, "tag": _tag(m)}
        for m in sorted(
            matched, key=lambda m: abs(m["raw_cosine"] - m["learned_mean"]), reverse=True
        )[:5]
    ]

    return {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "n_matched_pairs": len(matched),
        "top_agreements": top_agreements,
        "top_disagreements": top_disagreements,
    }


def plot_relation_ranking(ranking: list[dict[str, Any]], out_path: Path) -> None:
    """Bar chart, relations in the order given (rank_relations already
    sorts descending by mean), error bars = std."""
    relations = [row["relation"] for row in ranking]
    means = [row["mean"] for row in ranking]
    stds = [row["std"] for row in ranking]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(relations, means, yerr=stds, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean transferability (cosine similarity)")
    ax.set_title("Relation-wise transferability")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_class_pair_heatmap(pair_ranking: list[dict[str, Any]], out_path: Path) -> None:
    """Rows = class pairs, in the order given (rank_class_pairs already
    sorts descending by mean -- most-transferable pair at the top).
    Columns = relations, derived from whatever's present in pair_ranking,
    never hardcoded. A pair missing a relation renders NaN (blank cell),
    matching class_pair_summary's "omit, don't zero-fill" rule."""
    relations = sorted({relation for row in pair_ranking for relation in row["per_relation"]})
    row_labels = [f"{row['class_a']} vs {row['class_b']}" for row in pair_ranking]
    data = [
        [row["per_relation"].get(relation, float("nan")) for relation in relations]
        for row in pair_ranking
    ]

    fig, ax = plt.subplots(figsize=(1.6 * len(relations) + 2, 0.4 * len(row_labels) + 2))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(relations)))
    ax.set_xticklabels(relations, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if value == value:  # not NaN
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title("Class-pair transferability by relation")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run(
    run_dir: Path, raw_similarity_path: Path, num_tasks: int = NUM_TASKS
) -> dict[str, Any]:
    """Loads records, computes every summary/ranking/comparison, and
    returns the report dict with metadata populated except for
    "generated_at" and "git_commit" (left None -- see main()). Does not
    write anything; deterministic for identical inputs."""
    run_dir = Path(run_dir)
    records = load_transferability_records(run_dir, num_tasks=num_tasks)

    relation_summary = relation_wise_summary(records)
    relation_ranking = rank_relations(relation_summary)
    relation_evolution = relation_wise_task_evolution(records)

    pair_summary = class_pair_summary(records)
    pair_ranking = rank_class_pairs(pair_summary)

    raw_matrix = pd.read_csv(raw_similarity_path, index_col=0)
    raw_comparison = compare_to_raw_feature_similarity(pair_summary, raw_matrix)

    relations = sorted({record["relation"] for record in records})

    return {
        "metadata": {
            "source_run": str(run_dir),
            "generated_at": None,
            "git_commit": None,
            "similarity_matrix_path": str(raw_similarity_path),
            "num_tasks_analyzed": num_tasks,
            "num_records": len(records),
            "num_class_pairs": len(pair_summary),
            "num_relations": len(relations),
            "relations": relations,
        },
        "relation_ranking": relation_ranking,
        "relation_task_evolution": relation_evolution,
        "class_pair_ranking": pair_ranking,
        "raw_feature_comparison": raw_comparison,
    }


def write_report_json(report: dict[str, Any], out_path: Path) -> None:
    Path(out_path).write_text(json.dumps(report, indent=2))


def write_plots(report: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_relation_ranking(report["relation_ranking"], out_dir / "relation_ranking.png")
    plot_class_pair_heatmap(report["class_pair_ranking"], out_dir / "class_pair_heatmap.png")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Transferability analysis report (Step 10a).")
    parser.add_argument("--run-dir", default="runs/step4")
    parser.add_argument("--raw-similarity", default="data/similarity/similarity_matrix.csv")
    parser.add_argument("--out-dir", default="runs/step4/transferability_report")
    args = parser.parse_args()

    report = run(Path(args.run_dir), Path(args.raw_similarity))
    report["metadata"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["metadata"]["git_commit"] = _git_commit()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report_json(report, out_dir / "transferability_report.json")
    write_plots(report, out_dir)
    print(f"[transferability_report] wrote {out_dir}")


if __name__ == "__main__":
    main()
