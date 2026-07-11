"""Step 1 — Raw traffic preprocessing.

Turns the four raw NetFlow v2 CSVs into task-partitioned, sampled, split Parquet
tables ready for graph construction (Step 2). Implements the pipeline described
in ``docs/datasets.md`` §5.

Strategy (two streaming passes; files are up to 6 GB and never read whole):
  Pass 1  — read only the ``Attack`` column, count flows per canonical class and
            benign flows per dataset.
  Pass 2  — read full chunks; keep each attack row with probability
            min(1, cap / class_total) and each benign row with
            min(1, cap / dataset_benign_total). This yields ~uniform subsamples
            of ~cap per class while keeping every rare class in full.
Then per task: draw a fresh benign subset from the task's datasets, drop exact
duplicates, stratified train/val/test split, write ``task_{t}.parquet`` plus a
``manifest.json``.

Run:  trench-preprocess --config configs/preprocess.yaml
  or: python -m trench_ids.preprocess --config configs/preprocess.yaml
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from trench_ids.labels import (
    BENIGN,
    EXCLUDED_CLASSES,
    RAW_TO_CANONICAL,
    TASK_DATASETS,
    TASK_THEMES,
    UnknownAttackLabel,
    canonical_classes,
    task_of,
)

# Metadata columns this pipeline adds to every row (kept separate from the
# original NetFlow columns so dedup can operate on the original schema only).
META_COLS = ["source_dataset", "canonical_label", "task"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and lightly validate the preprocessing YAML config."""
    cfg = yaml.safe_load(Path(path).read_text())
    ratios = cfg["split"]
    total = ratios["train"] + ratios["val"] + ratios["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    return cfg


def _dataset_csv(raw_dir: Path, dir_name: str) -> Path:
    """Path to a dataset's payload CSV (``<name>/data/<name>.csv``)."""
    return raw_dir / dir_name / "data" / f"{dir_name}.csv"


def _map_canonical(attack: pd.Series) -> pd.Series:
    """Vectorized raw ``Attack`` -> canonical class, raising on any unknown."""
    mapped = attack.map(RAW_TO_CANONICAL)
    if mapped.isna().any():
        unknown = sorted(attack[mapped.isna()].unique())
        raise UnknownAttackLabel(
            f"Unmapped Attack label(s) {unknown}. Add them to RAW_TO_CANONICAL "
            f"in labels.py (and docs/datasets.md §4)."
        )
    return mapped


def pass1_counts(cfg: dict[str, Any]) -> tuple[Counter, Counter, list[str]]:
    """Count flows per canonical class and benign flows per dataset.

    Returns (attack_counts, benign_counts_per_dataset, original_columns).
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    chunk_size = cfg["sampling"]["chunk_size"]
    attack_counts: Counter = Counter()
    benign_counts: Counter = Counter()
    original_cols: list[str] = []

    for dir_name, code in cfg["datasets"].items():
        csv = _dataset_csv(raw_dir, dir_name)
        if not original_cols:
            original_cols = list(pd.read_csv(csv, nrows=0).columns)
        print(f"[pass1] counting {code} ({dir_name}) ...", flush=True)
        for chunk in pd.read_csv(csv, usecols=["Attack"], chunksize=chunk_size):
            canon = _map_canonical(chunk["Attack"])
            vc = canon.value_counts()
            benign_counts[code] += int(vc.get(BENIGN, 0))
            for cls, n in vc.items():
                if cls != BENIGN and cls not in EXCLUDED_CLASSES:
                    attack_counts[cls] += int(n)
    return attack_counts, benign_counts, original_cols


def pass2_sample(
    cfg: dict[str, Any],
    attack_counts: Counter,
    benign_counts: Counter,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Stream full rows, Bernoulli-subsample per class / per-dataset benign.

    Returns (all_attacks_df, {dataset_code: benign_pool_df}).
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    chunk_size = cfg["sampling"]["chunk_size"]
    attack_cap = cfg["sampling"]["attack_per_class_cap"]
    benign_cap = cfg["sampling"]["benign_per_dataset_cap"]

    # Keep-probabilities. Classes/datasets at or below the cap keep everything
    # (prob 1.0) — this is the per-class floor that protects rare classes.
    attack_prob = {c: min(1.0, attack_cap / n) for c, n in attack_counts.items()}
    benign_prob = {d: min(1.0, benign_cap / max(n, 1)) for d, n in benign_counts.items()}

    attack_parts: list[pd.DataFrame] = []
    benign_parts: dict[str, list[pd.DataFrame]] = {code: [] for code in cfg["datasets"].values()}

    for dir_name, code in cfg["datasets"].items():
        csv = _dataset_csv(raw_dir, dir_name)
        print(f"[pass2] sampling {code} ({dir_name}) ...", flush=True)
        for chunk in pd.read_csv(csv, chunksize=chunk_size):
            canon = _map_canonical(chunk["Attack"])
            # Drop excluded classes (e.g. Worms) before sampling.
            keep = ~canon.isin(EXCLUDED_CLASSES)
            if not keep.all():
                chunk = chunk.loc[keep].reset_index(drop=True)
                canon = canon.loc[keep].reset_index(drop=True)
            chunk = chunk.assign(source_dataset=code, canonical_label=canon.values)
            draw = rng.random(len(chunk))

            is_benign = canon.values == BENIGN
            # Attacks: per-row prob depends on class.
            atk = chunk[~is_benign]
            if len(atk):
                p = atk["canonical_label"].map(attack_prob).to_numpy()
                keep = draw[~is_benign] < p
                kept = atk[keep]
                if len(kept):
                    attack_parts.append(kept)
            # Benign: constant prob for the dataset.
            ben = chunk[is_benign]
            if len(ben):
                keep = draw[is_benign] < benign_prob[code]
                kept = ben[keep]
                if len(kept):
                    benign_parts[code].append(kept)

    attacks = (
        pd.concat(attack_parts, ignore_index=True) if attack_parts else pd.DataFrame()
    )
    attacks["task"] = attacks["canonical_label"].map(task_of).astype(int)
    # Trim any class that overshot its cap (Bernoulli variance), deterministically.
    attacks = _trim_per_group(attacks, "canonical_label", attack_cap, rng)

    benign_pools: dict[str, pd.DataFrame] = {}
    for code, parts in benign_parts.items():
        pool = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(pool) > benign_cap:
            pool = pool.sample(n=benign_cap, random_state=_seed_from(rng))
        benign_pools[code] = pool.reset_index(drop=True)
    return attacks.reset_index(drop=True), benign_pools


def _trim_per_group(
    df: pd.DataFrame, by: str, cap: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Downsample any group larger than ``cap`` to exactly ``cap`` rows."""
    if df.empty:
        return df
    keep_idx: list[np.ndarray] = []
    for _, grp in df.groupby(by, sort=False):
        idx = grp.index.to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep_idx.append(idx)
    return df.loc[np.concatenate(keep_idx)]


def _seed_from(rng: np.random.Generator) -> int:
    """Derive a fresh int seed from an rng (for pandas.sample reproducibility)."""
    return int(rng.integers(0, 2**31 - 1))


def _draw_benign_for_task(
    task: int,
    benign_pools: dict[str, pd.DataFrame],
    benign_per_task: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Draw a fresh benign subset for a task from its contributing datasets."""
    sources = TASK_DATASETS[task]
    per_ds = max(benign_per_task // len(sources), 1)
    drawn: list[pd.DataFrame] = []
    for code in sources:
        pool = benign_pools.get(code)
        if pool is None or pool.empty:
            continue
        n = min(per_ds, len(pool))
        drawn.append(pool.sample(n=n, random_state=_seed_from(rng)))
    if not drawn:
        return pd.DataFrame()
    out = pd.concat(drawn, ignore_index=True)
    out["task"] = task
    return out


def _stratified_split(
    df: pd.DataFrame, ratios: dict[str, float], rng: np.random.Generator
) -> pd.Series:
    """Assign train/val/test per row, stratified by canonical class."""
    split = pd.Series(index=df.index, dtype=object)
    for _, grp in df.groupby("canonical_label", sort=False):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * ratios["train"]))
        n_val = int(round(n * ratios["val"]))
        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train : n_train + n_val]] = "val"
        split.loc[idx[n_train + n_val :]] = "test"
    return split


def assemble_tasks(
    cfg: dict[str, Any],
    attacks: pd.DataFrame,
    benign_pools: dict[str, pd.DataFrame],
    original_cols: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build, split, and write one Parquet per task; return the manifest dict."""
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = cfg["split"]
    benign_per_task = cfg["sampling"]["benign_per_task"]
    do_dedup = cfg["dedup"]
    out_cols = original_cols + META_COLS + ["split"]

    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "sampling": cfg["sampling"],
        "split": ratios,
        "dedup": do_dedup,
        "tasks": {},
    }

    for task in sorted(TASK_THEMES):
        atk = attacks[attacks["task"] == task] if not attacks.empty else pd.DataFrame()
        ben = _draw_benign_for_task(task, benign_pools, benign_per_task, rng)
        frame = pd.concat([atk, ben], ignore_index=True)
        if frame.empty:
            continue

        if do_dedup:
            before = len(frame)
            frame = frame.drop_duplicates(subset=original_cols, ignore_index=True)
            dropped = before - len(frame)
        else:
            dropped = 0

        frame["split"] = _stratified_split(frame, ratios, rng)
        frame = frame[out_cols]

        path = out_dir / f"task_{task}.parquet"
        frame.to_parquet(path, index=False)

        manifest["tasks"][str(task)] = {
            "theme": TASK_THEMES[task],
            "datasets": list(TASK_DATASETS[task]),
            "n_rows": int(len(frame)),
            "duplicates_dropped": int(dropped),
            "class_counts": {k: int(v) for k, v in frame["canonical_label"].value_counts().items()},
            "split_counts": {k: int(v) for k, v in frame["split"].value_counts().items()},
            "source_counts": {k: int(v) for k, v in frame["source_dataset"].value_counts().items()},
            "file": path.name,
        }
        print(
            f"[assemble] task {task} ({TASK_THEMES[task]}): {len(frame):,} rows "
            f"-> {path.name} (dropped {dropped:,} dups)",
            flush=True,
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(config_path: str | Path) -> dict[str, Any]:
    """Execute the full Step 1 pipeline; returns the manifest."""
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["seed"])

    attack_counts, benign_counts, original_cols = pass1_counts(cfg)
    print(
        f"[pass1] done. attack classes: {len(attack_counts)}; "
        f"total attack flows: {sum(attack_counts.values()):,}; "
        f"benign per dataset: {dict(benign_counts)}",
        flush=True,
    )
    # Sanity: every canonical attack class we expect actually appeared.
    missing = set(canonical_classes()) - {BENIGN} - set(attack_counts)
    if missing:
        print(f"[pass1] WARNING: expected classes absent from data: {sorted(missing)}")

    attacks, benign_pools = pass2_sample(cfg, attack_counts, benign_counts, rng)
    print(
        f"[pass2] done. sampled attack flows: {len(attacks):,}; "
        f"benign pools: {{ {', '.join(f'{k}:{len(v):,}' for k, v in benign_pools.items())} }}",
        flush=True,
    )

    manifest = assemble_tasks(cfg, attacks, benign_pools, original_cols, rng)
    print(f"[done] wrote {len(manifest['tasks'])} task files to {cfg['paths']['out_dir']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="TRENCH-IDS Step 1 preprocessing.")
    parser.add_argument(
        "--config",
        default="configs/preprocess.yaml",
        help="Path to the preprocessing YAML config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
