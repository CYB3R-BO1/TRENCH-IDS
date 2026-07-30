"""Attack-class similarity analysis (professor-directed Step 1 redesign input).

For the 14-class candidate pool (docs/attack-class-counts.md), draws a random
sample per class and writes it to its own CSV. Method (per spec):
  1. Sample ~5000 rows per attack class, stored separately.
  2. Compute the mean feature vector of every attack, over the 37 flow-
     statistic features (docs/datasets.md SS2.1 -- the same feature set
     Step 2's Flow node will use), then standardize those per-class means
     (z-score across the class-mean vectors, not the underlying samples).
  3. Compute pairwise cosine similarity between the standardized class means.
Output feeds the task-redesign decision: which attack classes are "too
similar" to co-locate in the same continual-learning task.

Sampling: a count pass first gets the exact total per class, then a single
streaming pass draws an exact uniform sample of ``samples_per_class`` rows per
class without replacement -- every class ends up with exactly that many rows
(no shortfall), by pre-selecting which occurrence-index of each class to keep
and matching rows against that selection as the stream is scanned.

Run:  python -m trench_ids.similarity --config configs/similarity.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trench_ids.labels import CLASS_DATASETS, RAW_TO_CANONICAL, UnknownAttackLabel


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _dataset_csv(raw_dir: Path, dir_name: str) -> Path:
    return raw_dir / dir_name / "data" / f"{dir_name}.csv"


def _map_canonical(attack: pd.Series) -> pd.Series:
    mapped = attack.map(RAW_TO_CANONICAL)
    if mapped.isna().any():
        unknown = sorted(attack[mapped.isna()].unique())
        raise UnknownAttackLabel(f"Unmapped Attack label(s) {unknown}.")
    return mapped


def count_classes(
    cfg: dict[str, Any],
    classes: list[str],
    class_datasets: dict[str, set[str]] | None = None,
) -> Counter:
    """Count flows per candidate class, restricted to each class's allowed
    dataset(s) (streaming, Attack column only).

    class_datasets restricts which dataset(s) a class's rows may be counted
    from -- e.g. BoT-IoT also has DDoS/DoS-labeled rows, but those must not
    leak into the DDoS/DoS classes since BoT-IoT's only sanctioned
    contribution is Reconnaissance. Defaults to trench_ids.labels.CLASS_DATASETS.
    """
    class_datasets = class_datasets if class_datasets is not None else CLASS_DATASETS
    raw_dir = Path(cfg["paths"]["raw_dir"])
    chunk_size = cfg["chunk_size"]
    wanted = set(classes)
    counts: Counter = Counter()
    for dir_name, code in cfg["datasets"].items():
        csv = _dataset_csv(raw_dir, dir_name)
        print(f"[count] {dir_name} ...", flush=True)
        for chunk in pd.read_csv(csv, usecols=["Attack"], chunksize=chunk_size):
            canon = _map_canonical(chunk["Attack"])
            vc = canon.value_counts()
            for cls, n in vc.items():
                if cls in wanted and code in class_datasets.get(cls, set()):
                    counts[cls] += int(n)
    return counts


def sample_classes(
    cfg: dict[str, Any],
    classes: list[str],
    counts: Counter,
    rng: np.random.Generator,
    class_datasets: dict[str, set[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Exact uniform sample of ``samples_per_class`` rows per class, no replacement,
    restricted to each class's allowed dataset(s).

    Equivalent in result to reservoir sampling (a uniform random subset of
    fixed size drawn from a stream) but implemented via pre-selected
    occurrence indices, since the exact per-class total (already
    CLASS_DATASETS-restricted) is known from ``count_classes``. For each
    class, ``n_target`` of its ``total`` occurrences (0-indexed, in stream
    order *within its allowed datasets*) are chosen up front; as each
    allowed dataset's stream is scanned, a running per-class occurrence
    counter is compared against that selection.
    """
    class_datasets = class_datasets if class_datasets is not None else CLASS_DATASETS
    raw_dir = Path(cfg["paths"]["raw_dir"])
    chunk_size = cfg["chunk_size"]
    n_target = cfg["samples_per_class"]
    wanted = set(classes)
    selected: dict[str, np.ndarray] = {
        c: rng.choice(counts[c], size=min(n_target, counts[c]), replace=False) for c in classes
    }
    seen = dict.fromkeys(classes, 0)

    parts: dict[str, list[pd.DataFrame]] = {c: [] for c in classes}
    for dir_name, code in cfg["datasets"].items():
        csv = _dataset_csv(raw_dir, dir_name)
        print(f"[sample] {dir_name} ...", flush=True)
        allowed_here = {c for c in wanted if code in class_datasets.get(c, set())}
        if not allowed_here:
            continue
        for chunk in pd.read_csv(csv, chunksize=chunk_size):
            canon = _map_canonical(chunk["Attack"])
            mask = canon.isin(allowed_here)
            if not mask.any():
                continue
            sub = chunk.loc[mask].reset_index(drop=True)
            canon_sub = canon.loc[mask].reset_index(drop=True)

            keep_mask = np.zeros(len(sub), dtype=bool)
            for cls in canon_sub.unique():
                positions = np.flatnonzero(canon_sub.to_numpy() == cls)
                occurrence_idx = seen[cls] + np.arange(len(positions))
                keep_mask[positions[np.isin(occurrence_idx, selected[cls])]] = True
                seen[cls] += len(positions)

            kept = sub.loc[keep_mask].assign(
                source_dataset=code, canonical_label=canon_sub.loc[keep_mask].values
            )
            for cls, grp in kept.groupby("canonical_label"):
                parts[cls].append(grp)

    samples: dict[str, pd.DataFrame] = {}
    for cls in classes:
        frame = pd.concat(parts[cls], ignore_index=True) if parts[cls] else pd.DataFrame()
        samples[cls] = frame.reset_index(drop=True)
    return samples


def write_samples(samples: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls, frame in samples.items():
        path = out_dir / f"{cls.replace(' ', '_')}.csv"
        frame.to_csv(path, index=False)
        print(f"[write] {cls}: {len(frame)} rows -> {path.name}", flush=True)


def compute_centroids(
    samples: dict[str, pd.DataFrame], features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Mean feature vector per class first, then z-score standardize the means.

    Order matters here: this computes each class's raw mean feature vector,
    then standardizes across the resulting class-mean vectors (reference
    distribution = the N classes, not the underlying per-flow samples).
    Returns (centroids indexed by class, features actually used). Zero-variance
    features (identical mean across every class) are dropped -- z-scoring a
    constant column is undefined.
    """
    raw_means = pd.DataFrame({cls: df[features].mean() for cls, df in samples.items()}).T
    raw_means.index.name = "canonical_label"

    stats = raw_means.agg(["mean", "std"])
    zero_var = [f for f in features if stats.loc["std", f] == 0 or pd.isna(stats.loc["std", f])]
    used = [f for f in features if f not in zero_var]
    if zero_var:
        print(f"[centroids] dropping zero-variance features: {zero_var}", flush=True)

    mean = stats.loc["mean", used]
    std = stats.loc["std", used]
    centroids = (raw_means[used] - mean) / std
    return centroids, used


def cosine_similarity_matrix(centroids: pd.DataFrame) -> pd.DataFrame:
    vecs = centroids.to_numpy()
    normed = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = normed @ normed.T
    return pd.DataFrame(sim, index=centroids.index, columns=centroids.index)


def most_similar_pairs(sim: pd.DataFrame) -> pd.DataFrame:
    """Every unordered class pair, sorted by cosine similarity descending."""
    classes = list(sim.index)
    rows = [
        {"class_a": a, "class_b": b, "cosine_similarity": sim.loc[a, b]}
        for i, a in enumerate(classes)
        for b in classes[i + 1 :]
    ]
    return (
        pd.DataFrame(rows)
        .sort_values("cosine_similarity", ascending=False)
        .reset_index(drop=True)
    )


def run(config_path: str | Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    classes = cfg["classes"]
    features = cfg["features"]
    out_dir = Path(cfg["paths"]["out_dir"])
    rng = np.random.default_rng(cfg["seed"])

    print(f"[count] counting {len(classes)} candidate classes ...", flush=True)
    counts = count_classes(cfg, classes)
    missing = set(classes) - set(counts)
    if missing:
        raise ValueError(f"Candidate classes absent from data: {sorted(missing)}")
    print(f"[count] done: {dict(counts)}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "class_counts.json").write_text(json.dumps(dict(counts), indent=2))

    print(f"[sample] drawing ~{cfg['samples_per_class']} rows per class ...", flush=True)
    samples = sample_classes(cfg, classes, counts, rng)
    write_samples(samples, out_dir)

    centroids, used_features = compute_centroids(samples, features)
    centroids.to_csv(out_dir / "centroids.csv")

    sim = cosine_similarity_matrix(centroids)
    sim.to_csv(out_dir / "similarity_matrix.csv")

    pairs = most_similar_pairs(sim)
    pairs.to_csv(out_dir / "similarity_pairs.csv", index=False)
    print(f"[done] {len(used_features)} features used; outputs in {out_dir}", flush=True)
    print(pairs.to_string(index=False))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="TRENCH-IDS attack-class similarity analysis.")
    parser.add_argument("--config", default="configs/similarity.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
