"""Step 10 (third bullet) -- extraction of unseen-attack data.

Two evaluation datasets, both genuinely untouched by trench_ids.preprocess:

  Dataset A (evaluation_type="seen_class_unseen_samples") -- new samples of
    attack classes the model already knows (Reconnaissance/DoS from
    NF-UNSW-NB15-v2, DDoS/DoS from NF-BoT-IoT-v2). trench_ids.preprocess
    keeps 100% of every CLASS_DATASETS-allowed attack row for the 3
    in-scope datasets, so there is no leftover untouched subset of an
    already-selected class within those datasets -- these rows come from
    sources the training pipeline never opens at all (NF-UNSW-NB15-v2) or
    filters out entirely via CLASS_DATASETS (BoT-IoT's DDoS/DoS).

  Dataset B (evaluation_type="unseen_class") -- attack classes with no
    CLASS_DATASETS entry at all (Backdoor, MITM, Ransomware, Web Attacks,
    Theft). Their raw labels map successfully via RAW_TO_CANONICAL, but
    trench_ids.preprocess._class_allowed always returns False for them, so
    they are read then discarded in every run of the frozen pipeline.

This module deliberately bypasses _class_allowed/CLASS_DATASETS (the
opposite of trench_ids.preprocess) -- it does not modify or call that
check. It reuses _map_canonical and _drop_corrupted_rows for row-level
consistency with Step 1.

Run: trench-unseen-data --config configs/unseen.yaml
  or: python -m trench_ids.unseen_data --config configs/unseen.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trench_ids.labels import RAW_TO_CANONICAL
from trench_ids.preprocess import _dataset_csv, _drop_corrupted_rows, _map_canonical


def extract_unseen_class(
    raw_dir: Path,
    dataset_dir: str,
    dataset_code: str,
    canonical_label: str,
    chunk_size: int,
    sample_cap: int | None,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream `dataset_dir`'s CSV; keep only rows whose canonical label
    equals `canonical_label`, regardless of CLASS_DATASETS allowance.

    Applies _drop_corrupted_rows (same NaN/inf/float32-overflow filter as
    Step 1). If more than `sample_cap` matching rows exist, draws a seeded
    uniform sample of exactly `sample_cap` rows (for high-volume classes
    like BoT-IoT's DDoS/DoS, 18.3M/16.7M available); otherwise keeps every
    matching row. Returns (frame, manifest_entry) -- manifest_entry has no
    "evaluation_type"/"used_in_training"/"output_path" keys yet; run()
    adds those since they're a property of the config entry, not of the
    extraction itself.
    """
    csv_path = _dataset_csv(raw_dir, dataset_dir)
    original_cols = list(pd.read_csv(csv_path, nrows=0).columns)

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Pre-filter to only rows with raw Attack labels in RAW_TO_CANONICAL
        # to avoid UnknownAttackLabel errors on unmapped labels (e.g., UNSW-NB15's
        # Exploits, Fuzzers, etc. which aren't used for training).
        chunk = chunk[chunk["Attack"].isin(RAW_TO_CANONICAL)]
        if chunk.empty:
            continue
        canon = _map_canonical(chunk["Attack"])
        match = (canon == canonical_label).to_numpy()
        if not match.any():
            continue
        matched = chunk.loc[match].copy()
        matched["flow_id"] = [f"{dataset_code}-{i}" for i in chunk.index[match]]
        matched["source_dataset"] = dataset_code
        matched["canonical_label"] = canonical_label
        parts.append(matched)

    if parts:
        frame = pd.concat(parts, ignore_index=True)
    else:
        frame = pd.DataFrame(
            columns=original_cols + ["flow_id", "source_dataset", "canonical_label"]
        )

    rows_found = len(frame)
    frame, corrupted_dropped = _drop_corrupted_rows(frame, original_cols)

    sampled = False
    if sample_cap is not None and len(frame) > sample_cap:
        seed = int(rng.integers(0, 2**31 - 1))
        frame = frame.sample(n=sample_cap, random_state=seed).reset_index(drop=True)
        sampled = True

    manifest_entry = {
        "canonical_label": canonical_label,
        "dataset": dataset_dir,
        "dataset_code": dataset_code,
        "rows_found": rows_found,
        "corrupted_rows_dropped": corrupted_dropped,
        "rows_kept": len(frame),
        "sampled": sampled,
        "sample_cap": sample_cap,
    }
    return frame, manifest_entry


def _slug(dataset_code: str, canonical_label: str) -> str:
    return f"{dataset_code}_{canonical_label.lower().replace(' ', '_')}"


def run(config_path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    raw_dir = Path(cfg["paths"]["raw_dir"])
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    chunk_size = cfg["chunk_size"]
    sample_cap = cfg.get("sample_cap")

    manifest: dict[str, Any] = {"seed": cfg["seed"], "classes": {}}
    for entry in cfg["classes"]:
        canonical_label = entry["canonical_label"]
        dataset_dir = entry["dataset_dir"]
        dataset_code = entry["dataset_code"]
        evaluation_type = entry["evaluation_type"]

        frame, manifest_entry = extract_unseen_class(
            raw_dir, dataset_dir, dataset_code, canonical_label,
            chunk_size, sample_cap, rng,
        )

        slug = _slug(dataset_code, canonical_label)
        out_path = out_dir / f"{slug}.parquet"
        frame.to_parquet(out_path, index=False)

        manifest_entry["used_in_training"] = False
        manifest_entry["evaluation_type"] = evaluation_type
        manifest_entry["output_path"] = str(out_path)
        manifest["classes"][slug] = manifest_entry

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract unseen-attack rows for Step 10 inference."
    )
    parser.add_argument("--config", default="configs/unseen.yaml")
    args = parser.parse_args()
    manifest = run(args.config)
    for slug, entry in manifest["classes"].items():
        print(f"{slug}: {entry['rows_kept']} rows kept ({entry['evaluation_type']})")


if __name__ == "__main__":
    main()
