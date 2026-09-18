"""Step 1 (No-Cap) — Raw traffic preprocessing.

Variant of ``preprocess.py`` that uses **all eligible attack rows** (no per-task
quota cap). Benign samples are drawn disjointly toward a target
attack:benign ratio (default 3:1); when a task's benign pool is exhausted,
all available benign rows are used and the realized ratio is reported.

ISOLATED from the main benchmark: outputs to ``data/no-cap/processed/``.
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
import psutil
import yaml

from trench_ids.labels import (
    BENIGN,
    CLASS_DATASETS,
    RAW_TO_CANONICAL,
    TASK_DATASETS,
    TASK_THEMES,
    UnknownAttackLabel,
    attack_classes_for_task,
    canonical_classes,
    task_of,
)

# Metadata columns this pipeline adds to every row (kept separate from the
# original NetFlow columns so dedup can operate on the original schema only).
#
# ``source_row`` is the row's 0-indexed line number in its own dataset CSV.
# It is the same integer already embedded in ``flow_id``, promoted to a real
# int column because Step 2 needs it as a *sort key*: mini-graphs are chunked
# in capture order (``graphs.py``), which requires ordering rows by
# (source_dataset, source_row) rather than re-parsing it out of a string on
# every build.
META_COLS = ["flow_id", "source_dataset", "source_row", "canonical_label", "task"]

# Anything beyond float32's range is a legitimate float64 value in the raw
# CSV but silently becomes `inf` when Step 2 (graphs.py) casts Flow features
# to float32 -- filtered out here so it never reaches a saved graph.
FLOAT32_MAX = float(np.finfo(np.float32).max)


def _log_rss(label: str) -> None:
    """Print this process' current resident memory -- diagnostic only."""
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    print(f"[mem] {label}: {rss_gb:.2f} GiB RSS", flush=True)


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


def _class_allowed(canonical: str, code: str) -> bool:
    """Whether a canonical class's rows may be drawn from this dataset code."""
    allowed = CLASS_DATASETS.get(canonical)
    return allowed is not None and code in allowed


def pass1_counts(cfg: dict[str, Any]) -> tuple[Counter, Counter, list[str]]:
    """Count flows per (canonical class, dataset) and benign flows per dataset.

    Attack counts are keyed by ``(canonical_class, dataset_code)`` rather
    than by class alone. Only ``CLASS_DATASETS``-sanctioned (class, dataset)
    pairs are counted.

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
                if cls != BENIGN and _class_allowed(cls, code):
                    attack_counts[(cls, code)] += int(n)
    return attack_counts, benign_counts, original_cols


def compute_quotas_no_cap(
    attack_counts: Counter,
    benign_counts: Counter,
    attack_benign_ratio: float,
) -> dict[str, Any]:
    """Plan row quotas WITHOUT per-task attack cap.

    Every task gets ALL available attack rows for its classes (summed across
    sanctioned datasets). Benign target per task = attack_rows / attack_benign_ratio.
    Benign allocated disjointly per task from that task's own datasets.
    When benign pool exhausted, all available benign used and realized ratio reported.

    Returns dict with:
      - attack_quota: (class, dataset) -> rows (ALL available)
      - attack_per_task_by_class: task -> {class: count}
      - benign_alloc: task -> {dataset: rows_drawn}
      - benign_quota: dataset -> total_rows_drawn
      - benign_target_per_task: task -> target_rows
      - benign_available_per_task: task -> available_rows
      - attack_benign_ratio: target ratio
    """
    attack_quota: dict[tuple[str, str], int] = {}
    per_task_attack: dict[int, dict[str, int]] = {}
    benign_target_per_task: dict[int, int] = {}

    for task in sorted(TASK_THEMES):
        classes = attack_classes_for_task(task)
        by_class = {}
        for cls in classes:
            total = sum(
                attack_counts.get((cls, ds), 0)
                for ds in sorted(CLASS_DATASETS[cls])
            )
            by_class[cls] = total
            for ds in sorted(CLASS_DATASETS[cls]):
                n = attack_counts.get((cls, ds), 0)
                if n:
                    attack_quota[(cls, ds)] = n
        per_task_attack[task] = by_class
        attack_rows = sum(by_class.values())
        target_benign = max(1, round(attack_rows / attack_benign_ratio))
        benign_target_per_task[task] = target_benign

    # Benign allocation: disjoint per task from task's own datasets
    benign_alloc: dict[int, dict[str, int]] = {}
    remaining_benign = {ds: int(n) for ds, n in benign_counts.items()}

    for task in sorted(TASK_THEMES):
        sources = TASK_DATASETS[task]
        capacity = {ds: remaining_benign.get(ds, 0) for ds in sources}
        target = benign_target_per_task[task]

        # Waterfill up to target, never exceeding available capacity
        drawn = _waterfill_no_cap(capacity, target)
        benign_alloc[task] = drawn
        for ds, n in drawn.items():
            remaining_benign[ds] = remaining_benign.get(ds, 0) - n

    benign_quota: dict[str, int] = {}
    for drawn in benign_alloc.values():
        for ds, n in drawn.items():
            benign_quota[ds] = benign_quota.get(ds, 0) + n

    # Compute available benign per task for reporting
    benign_available_per_task: dict[int, int] = {}
    for task in sorted(TASK_THEMES):
        sources = TASK_DATASETS[task]
        avail = sum(benign_counts.get(ds, 0) for ds in sources)
        benign_available_per_task[task] = avail

    return {
        "attack_quota": attack_quota,
        "attack_per_task_by_class": per_task_attack,
        "benign_alloc": benign_alloc,
        "benign_quota": benign_quota,
        "benign_target_per_task": benign_target_per_task,
        "benign_available_per_task": benign_available_per_task,
        "attack_benign_ratio": attack_benign_ratio,
    }


def _waterfill_no_cap(capacity: dict[Any, int], budget: int) -> dict[Any, int]:
    """Split ``budget`` as evenly as possible across keys, never exceeding capacity.

    Same logic as main ``_waterfill`` but used here for benign allocation.
    """
    alloc = {k: 0 for k in capacity}
    remaining = int(budget)
    open_keys = [k for k, c in capacity.items() if c > 0]
    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        if share == 0:
            for k in open_keys[:extra]:
                alloc[k] += 1
            remaining = 0
            break
        newly_saturated = []
        for k in open_keys:
            take = min(share, capacity[k] - alloc[k])
            alloc[k] += take
            remaining -= take
            if alloc[k] >= capacity[k]:
                newly_saturated.append(k)
        if not newly_saturated:
            continue
        open_keys = [k for k in open_keys if k not in newly_saturated]
    return alloc


def _choose_ordinals(total: int, want: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted 0-indexed positions of the ``want`` rows to keep out of ``total``."""
    if want >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=want, replace=False)).astype(np.int64)


def _ordinal_mask(chosen: np.ndarray, ordinals: np.ndarray) -> np.ndarray:
    """Which of ``ordinals`` are present in the sorted array ``chosen``."""
    if chosen.size == 0 or ordinals.size == 0:
        return np.zeros(ordinals.shape, dtype=bool)
    pos = np.searchsorted(chosen, ordinals)
    np.clip(pos, 0, chosen.size - 1, out=pos)
    return chosen[pos] == ordinals


def pass2_sample(
    cfg: dict[str, Any],
    attack_counts: Counter,
    benign_counts: Counter,
    quotas: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[dict[int, Path | None], dict[str, pd.DataFrame], Counter]:
    """Stream full rows a second time, keeping exactly the planned quota.

    For attack rows: keep ALL available rows (no sampling), write per-task parquet files incrementally.
    For benign rows: keep exactly the allocated quota per dataset.
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    chunk_size = cfg["sampling"]["chunk_size"]
    attack_quota: dict[tuple[str, str], int] = quotas["attack_quota"]
    benign_quota: dict[str, int] = quotas["benign_quota"]

    # For benign: pre-select ordinals. For attack: we want ALL rows, so no ordinal selection needed.
    chosen_benign = {
        code: _choose_ordinals(int(benign_counts.get(code, 0)), want, rng)
        for code, want in benign_quota.items()
    }
    seen_benign: Counter = Counter()

    # Track per-task attack parts for incremental writing
    attack_parts_by_task: dict[int, list[pd.DataFrame]] = {t: [] for t in sorted(TASK_THEMES)}
    # Track total rows per task for manifest reporting
    task_attack_counts: Counter = Counter()
    benign_parts: dict[str, list[pd.DataFrame]] = {code: [] for code in cfg["datasets"].values()}

    _log_rss("pass2: start")
    for dir_name, code in cfg["datasets"].items():
        csv = _dataset_csv(raw_dir, dir_name)
        print(f"[pass2] streaming {code} ({dir_name}) ...", flush=True)
        for chunk in pd.read_csv(csv, chunksize=chunk_size):
            canon = _map_canonical(chunk["Attack"]).to_numpy()
            row_no = chunk.index.to_numpy()

            for cls in np.unique(canon):
                key = (cls, code)
                if cls == BENIGN:
                    if code not in chosen_benign:
                        continue
                    chosen = chosen_benign[code]
                    seen = seen_benign
                    seen_key: Any = code
                elif key in attack_quota:
                    # Attack: keep ALL rows (no sub-selection)
                    chosen = None
                    seen = None
                    seen_key = None
                else:
                    continue  # class not sanctioned for this dataset, or zero quota
                where = np.flatnonzero(canon == cls)
                if cls == BENIGN:
                    ordinals = seen[seen_key] + np.arange(where.size, dtype=np.int64)
                    seen[seen_key] += where.size
                    keep = where[_ordinal_mask(chosen, ordinals)]
                else:
                    keep = where  # KEEP ALL ATTACK ROWS
                if keep.size == 0:
                    continue
                kept = chunk.iloc[keep].assign(
                    flow_id=[f"{code}-{i}" for i in row_no[keep]],
                    source_dataset=code,
                    source_row=row_no[keep],
                    canonical_label=cls,
                )
                if cls == BENIGN:
                    benign_parts[code].append(kept)
                else:
                    task_id = task_of[cls]
                    attack_parts_by_task[task_id].append(kept)

    _log_rss("pass2: streaming complete, writing per-task attack parquets")
    # Write per-task attack parquets incrementally
    attack_parquet_paths: dict[int, Path | None] = {}
    out_dir = Path(cfg["paths"]["out_dir"])
    for task_id in sorted(TASK_THEMES):
        parts = attack_parts_by_task.get(task_id, [])
        if parts:
            task_df = pd.concat(parts, ignore_index=True)
            task_df["task"] = task_id
            path = out_dir / f"task_{task_id}_attacks.parquet"
            task_df.to_parquet(path, index=False)
            attack_parquet_paths[task_id] = path
            del parts, task_df  # free memory
        else:
            attack_parquet_paths[task_id] = None
    _log_rss("pass2: after attack parquet writes")

    # Benign pools (unchanged - small enough)
    benign_pools: dict[str, pd.DataFrame] = {}
    for code, parts in benign_parts.items():
        benign_pools[code] = (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        )
    _log_rss("pass2: after benign_pools concat")

    # Compute task_attack_counts from the parts we just wrote
    task_attack_counts: Counter = Counter()
    for task_id, path in attack_parquet_paths.items():
        if path is not None:
            df = pd.read_parquet(path)
            task_attack_counts[task_id] = len(df)
    return attack_parquet_paths, benign_pools, task_attack_counts


def _seed_from(rng: np.random.Generator) -> int:
    """Derive a fresh int seed from an rng (for pandas.sample reproducibility)."""
    return int(rng.integers(0, 2**31 - 1))


class BenignAllocator:
    """Hands each task a *disjoint* slice of every dataset's benign pool."""

    def __init__(self, benign_pools: dict[str, pd.DataFrame], rng: np.random.Generator):
        self._pools = {
            code: pool.sample(frac=1, random_state=_seed_from(rng)).reset_index(drop=True)
            for code, pool in benign_pools.items()
            if pool is not None and not pool.empty
        }
        self._cursor: dict[str, int] = {code: 0 for code in self._pools}

    def draw(self, task: int, alloc: dict[str, int]) -> pd.DataFrame:
        """Take this task's planned slice; ``alloc`` maps dataset code -> rows."""
        drawn: list[pd.DataFrame] = []
        for code, want in sorted(alloc.items()):
            pool = self._pools.get(code)
            if pool is None or want <= 0:
                continue
            start = self._cursor[code]
            stop = min(start + want, len(pool))
            if stop > start:
                drawn.append(pool.iloc[start:stop])
            self._cursor[code] = stop
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


def _drop_corrupted_rows(
    frame: pd.DataFrame, original_cols: list[str]
) -> tuple[pd.DataFrame, int]:
    """Drop rows with a non-finite numeric value, or a finite value that
    would overflow float32 at Step 2's feature cast -- plus rows with a
    missing categorical value.

    Memory-efficient version: checks columns individually to avoid
    the select_dtypes consolidation that OOMs on 5M+ row frames.
    """
    bad = np.zeros(len(frame), dtype=bool)

    # Check numeric columns one by one
    for col in original_cols:
        if col not in frame.columns:
            continue
        s = frame[col]
        if np.issubdtype(s.dtype, np.number):
            vals = s.to_numpy(dtype=np.float64)
            bad |= (~np.isfinite(vals) | (np.abs(vals) > FLOAT32_MAX))
        elif s.dtype == "object" or str(s.dtype).startswith("category"):
            bad |= s.isna().to_numpy()

    dropped = int(bad.sum())
    clean = frame.loc[~bad]
    # Avoid reset_index(drop=True) which doubles peak memory --
    # reassign RangeIndex directly (same trick as main preprocess.py)
    clean.index = pd.RangeIndex(len(clean))
    return clean, dropped


def assemble_tasks(
    cfg: dict[str, Any],
    attack_parquet_paths: dict[int, Path | None],
    benign_pools: dict[str, pd.DataFrame],
    original_cols: list[str],
    quotas: dict[str, Any],
    rng: np.random.Generator,
    task_attack_counts: Counter,
) -> dict[str, Any]:
    """Build, split, and write one Parquet per task; return the manifest dict."""
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = cfg["split"]
    do_dedup = cfg["dedup"]
    out_cols = original_cols + META_COLS + ["split"]
    allocator = BenignAllocator(benign_pools, rng)

    attack_benign_ratio = quotas["attack_benign_ratio"]

    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "sampling": cfg["sampling"],
        "split": ratios,
        "dedup": do_dedup,
        "quota_plan": {
            "attack_benign_ratio": attack_benign_ratio,
            "attack_quota": {f"{c}|{d}": n for (c, d), n in sorted(quotas["attack_quota"].items())},
            "benign_alloc": {str(t): a for t, a in quotas["benign_alloc"].items()},
            "benign_target_per_task": quotas["benign_target_per_task"],
            "benign_available_per_task": quotas["benign_available_per_task"],
        },
        "tasks": {},
    }

    for task in sorted(TASK_THEMES):
        path = attack_parquet_paths.get(task)
        atk = pd.read_parquet(path) if path is not None else pd.DataFrame()
        ben = allocator.draw(task, quotas["benign_alloc"].get(task, {}))
        frame = pd.concat([atk, ben], ignore_index=True)
        if frame.empty:
            continue

        frame, corrupted_dropped = _drop_corrupted_rows(frame, original_cols)

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

        # Compute realized attack:benign ratio for this task
        attack_rows = int((frame["canonical_label"] != BENIGN).sum())
        benign_rows = int((frame["canonical_label"] == BENIGN).sum())
        realized_ratio = attack_rows / max(benign_rows, 1)

        manifest["tasks"][str(task)] = {
            "theme": TASK_THEMES[task],
            "datasets": list(TASK_DATASETS[task]),
            "n_rows": int(len(frame)),
            "duplicates_dropped": int(dropped),
            "corrupted_rows_dropped": int(corrupted_dropped),
            "attack_rows": attack_rows,
            "benign_rows": benign_rows,
            "realized_attack_benign_ratio": float(realized_ratio),
            "attack_benign_ratio_target": attack_benign_ratio,
            "benign_pool_source": list(TASK_DATASETS[task]),
            "benign_pool_exhausted": benign_rows < quotas["benign_target_per_task"].get(task, 0),
            "class_counts": {k: int(v) for k, v in frame["canonical_label"].value_counts().items()},
            "split_counts": {k: int(v) for k, v in frame["split"].value_counts().items()},
            "source_counts": {k: int(v) for k, v in frame["source_dataset"].value_counts().items()},
            "file": path.name,
        }
        print(
            f"[assemble] task {task} ({TASK_THEMES[task]}): {len(frame):,} rows "
            f"(attack={attack_rows:,}, benign={benign_rows:,}, ratio={realized_ratio:.2f}:1) "
            f"-> {path.name} (dropped {dropped:,} dups, {corrupted_dropped:,} corrupted)",
            flush=True,
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(config_path: str | Path) -> dict[str, Any]:
    """Execute the full Step 1 (No-Cap) pipeline; returns the manifest."""
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["seed"])

    attack_counts, benign_counts, original_cols = pass1_counts(cfg)
    print(
        f"[pass1] done. (class, dataset) pairs: {len(attack_counts)}; "
        f"total available attack flows: {sum(attack_counts.values()):,}; "
        f"benign per dataset: {dict(benign_counts)}",
        flush=True,
    )

    quotas = compute_quotas_no_cap(
        attack_counts,
        benign_counts,
        cfg["sampling"]["attack_benign_ratio"],
    )
    for task in sorted(TASK_THEMES):
        by_class = quotas["attack_per_task_by_class"][task]
        benign_target = quotas["benign_target_per_task"][task]
        benign_available = quotas["benign_available_per_task"][task]
        benign_drawn = sum(quotas["benign_alloc"][task].values())
        print(
            f"[quota] task {task} ({TASK_THEMES[task]}): "
            f"attack {sum(by_class.values()):,} {by_class}, "
            f"benign target={benign_target:,}, available={benign_available:,}, "
            f"drawn={benign_drawn:,} (ratio={sum(by_class.values())/max(benign_drawn,1):.2f}:1)",
            flush=True,
        )

    attack_parquet_paths, benign_pools, task_attack_counts = pass2_sample(
        cfg, attack_counts, benign_counts, quotas, rng
    )
    print(
        f"[pass2] done. attack parquet paths: {list(attack_parquet_paths.keys())}; "
        f"benign pools: {{ {', '.join(f'{k}:{len(v):,}' for k, v in benign_pools.items())} }}",
        flush=True,
    )

    manifest = assemble_tasks(
        cfg, attack_parquet_paths, benign_pools, original_cols, quotas, rng, task_attack_counts
    )
    print(f"[done] wrote {len(manifest['tasks'])} task files to {cfg['paths']['out_dir']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="TRENCH-IDS Step 1 No-Cap preprocessing.")
    parser.add_argument(
        "--config",
        default="configs/no-cap/preprocess.yaml",
        help="Path to the preprocessing YAML config.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()