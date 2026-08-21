"""Step 1 — Raw traffic preprocessing.

Turns the three raw NetFlow v2 CSVs (NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2,
NF-BoT-IoT-v2 -- NF-UNSW-NB15-v2 stays excluded, see CLAUDE.md) into
task-partitioned, split Parquet tables ready for graph construction (Step 2).
Implements the pipeline described in ``docs/datasets.md`` §5 and
``docs/superpowers/specs/2026-07-13-dataset-task-respec-design.md`` §3.

Pipeline: load -> assign flow_id -> merge -> clean -> split -> persist.

  Load     — stream each dataset's CSV in chunks (files are up to 6 GB and
             never read whole).
  Pass 1 (pass1_counts) counts flows per (canonical class, dataset) --
             restricted to each class's sanctioned dataset(s) via
             ``CLASS_DATASETS`` (e.g. BoT-IoT rows labeled DDoS/DoS are not
             counted, since only Reconnaissance is sanctioned for BoT-IoT) --
             and counts benign flows per dataset (unrestricted).
  Plan (compute_quotas) turns those exact counts into a sampling plan: each
             task gets the same ``attack_per_task`` + ``benign_per_task``
             budget, waterfilled evenly across its classes and each class's
             source datasets, with each task's benign slice disjoint from
             every other task's. See ``compute_quotas`` for the three
             measured defects of the previous "keep every attack row, use a
             small fixed benign pool" design that this replaces.
  Assign flow_id / Merge (pass2_sample) streams full rows a second time and
             keeps exactly the planned quota: the row *positions* to keep
             are drawn up front per key, so realised counts are exact and
             peak memory is bounded by the plan (~3.1M rows) rather than by
             the raw data (~15.7M attack rows). Every kept row is stamped
             with a stable ``flow_id``
             (f"{dataset_code}-{original_csv_row_number}", 0-indexed) and a
             matching int ``source_row``, so identity survives to the Step 2
             Flow graph node and capture order stays recoverable. Attack rows
             across all three datasets are then concatenated (merged) into
             one frame and each row's task is assigned from its canonical
             class.
  Clean / Split (assemble_tasks) — per task: take that task's planned,
             disjoint benign slice (``BenignAllocator``), concat with that
             task's attack rows, drop exact duplicates on the original
             NetFlow schema (clean), stratified train/val/test split by
             canonical class, then persist as ``task_{t}.parquet`` plus a
             combined ``manifest.json``.

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
    """Print this process' current resident memory -- diagnostic only, added
    2026-08-11 after pass2_sample OOM'd on the raised benign_per_dataset_cap
    (see preprocess.yaml). Cheap enough to leave in permanently."""
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
    """Whether a canonical class's rows may be drawn from this dataset code.

    See trench_ids.labels.CLASS_DATASETS -- a class can appear as a raw
    label in an unsanctioned dataset (e.g. BoT-IoT also has DDoS/DoS rows)
    without being allowed to draw from it.
    """
    allowed = CLASS_DATASETS.get(canonical)
    return allowed is not None and code in allowed


def pass1_counts(cfg: dict[str, Any]) -> tuple[Counter, Counter, list[str]]:
    """Count flows per (canonical class, dataset) and benign flows per dataset.

    Attack counts are keyed by ``(canonical_class, dataset_code)`` rather
    than by class alone, because the quota planner
    (:func:`compute_quotas`) has to split a multi-source class's quota
    across its sources by availability -- e.g. ``DDoS`` may draw from ToN
    *and* CSE, and how much each can supply is only knowable per dataset.
    Only ``CLASS_DATASETS``-sanctioned (class, dataset) pairs are counted.

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


def _waterfill(capacity: dict[Any, int], budget: int) -> dict[Any, int]:
    """Split ``budget`` as evenly as possible across keys, never exceeding
    each key's ``capacity``.

    Keys that cannot absorb an equal share are saturated at their capacity
    and their unused remainder is redistributed among the rest, repeating
    until either the budget is exhausted or every key is saturated. This is
    what makes the benchmark's per-task class balance *as balanced as the
    data allows*: in T3 (DDoS + Infiltration) Infiltration saturates at its
    116,361 available rows and DDoS absorbs the remainder, instead of both
    classes being sampled at their wildly unequal natural proportions
    (~30:1), which left Infiltration at ~0.5% of the training signal and
    unmeasurable for per-class forgetting.

    Returns a dict over the same keys (0 for keys with no capacity). The
    allocated total is ``min(budget, sum(capacity))``.
    """
    alloc = {k: 0 for k in capacity}
    remaining = int(budget)
    open_keys = [k for k, c in capacity.items() if c > 0]
    while remaining > 0 and open_keys:
        share, extra = divmod(remaining, len(open_keys))
        if share == 0:
            # Fewer units left than open keys -- hand out one each, in a
            # deterministic order, until they run out.
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
            # Everyone absorbed a full share; only `extra` units are left,
            # which the share==0 branch will hand out on the next pass.
            continue
        open_keys = [k for k in open_keys if k not in newly_saturated]
    return alloc


def compute_quotas(
    attack_counts: Counter,
    benign_counts: Counter,
    attack_per_task: int,
    benign_per_task: int,
) -> dict[str, Any]:
    """Plan exactly how many rows to draw per (class, dataset) and per
    (task, dataset) for benign, before a single full row is read.

    Two properties this planner exists to guarantee, both of which the
    previous "keep every attack row, subsample benign to a small fixed
    pool" design violated:

    1. **No benign reuse.** Every task's benign quota is drawn from a
       *disjoint* slice of its datasets' benign rows, and the quota is set
       so Step 2's ``benign_ratio`` can be met without sampling with
       replacement. Previously ~5,600 unique benign train flows were
       replicated 54-158x to fill the ratio, so ~25% of every reported
       metric rested on a few thousand distinct flows, and 47-72 flow_ids
       leaked from one task's train split into another task's test split.
    2. **Equal task sizes and near-equal class sizes.** Every task gets the
       same ``attack_per_task`` + ``benign_per_task`` budget, waterfilled
       across its classes and their sources. Task size therefore stops
       being a confound in any cross-task forgetting comparison (the
       previous benchmark's largest task was 2.9x its smallest).

    Tasks are planned in ascending order so benign allocation is
    deterministic; a dataset that runs short (NF-BoT-IoT-v2 holds only
    ~135K benign rows in total) simply allocates what it has, and the
    shortfall is reported in the manifest rather than silently backfilled.
    """
    attack_quota: dict[tuple[str, str], int] = {}
    per_task_attack: dict[int, dict[str, int]] = {}
    for task in sorted(TASK_THEMES):
        classes = attack_classes_for_task(task)
        capacity = {
            cls: sum(attack_counts.get((cls, ds), 0) for ds in sorted(CLASS_DATASETS[cls]))
            for cls in classes
        }
        by_class = _waterfill(capacity, attack_per_task)
        per_task_attack[task] = by_class
        for cls, want in by_class.items():
            ds_capacity = {
                ds: attack_counts.get((cls, ds), 0) for ds in sorted(CLASS_DATASETS[cls])
            }
            for ds, n in _waterfill(ds_capacity, want).items():
                if n:
                    attack_quota[(cls, ds)] = n

    benign_alloc: dict[int, dict[str, int]] = {}
    remaining_benign = {ds: int(n) for ds, n in benign_counts.items()}
    for task in sorted(TASK_THEMES):
        sources = TASK_DATASETS[task]
        capacity = {ds: remaining_benign.get(ds, 0) for ds in sources}
        drawn = _waterfill(capacity, benign_per_task)
        benign_alloc[task] = drawn
        for ds, n in drawn.items():
            # .get default covers a dataset that TASK_DATASETS references but
            # this run doesn't include (e.g. a cut-down config); such a
            # dataset has zero capacity, so n is always 0 here.
            remaining_benign[ds] = remaining_benign.get(ds, 0) - n

    benign_quota: dict[str, int] = {}
    for drawn in benign_alloc.values():
        for ds, n in drawn.items():
            benign_quota[ds] = benign_quota.get(ds, 0) + n

    return {
        "attack_quota": attack_quota,
        "attack_per_task_by_class": per_task_attack,
        "benign_quota": benign_quota,
        "benign_alloc": benign_alloc,
        "attack_per_task": attack_per_task,
        "benign_per_task": benign_per_task,
    }


def _choose_ordinals(total: int, want: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted 0-indexed positions of the ``want`` rows to keep out of ``total``.

    Selection is planned up front from Pass 1's exact counts rather than
    done with a per-row Bernoulli draw, so the realised count is *exactly*
    the quota (a Bernoulli pass overshoots or undershoots by a random
    amount, which would silently unbalance the tasks this design is trying
    to balance) and peak memory stays at one int64 array per key instead of
    a growing list of kept chunks.
    """
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
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Stream full rows a second time, keeping exactly the planned quota.

    For every sanctioned ``(class, dataset)`` pair, and for every dataset's
    benign rows, the *positions* of the rows to keep are drawn up front from
    Pass 1's exact counts (:func:`_choose_ordinals`); this pass just walks
    the CSV maintaining a per-key running ordinal and keeps the rows whose
    ordinal was selected. Peak memory is therefore bounded by the quota
    (~3.1M rows across all six tasks) rather than by the raw data (~15.7M
    attack rows alone, which is what OOM'd this machine at 16 GB once the
    benign pool was scaled up to match).

    Each kept row gets a stable ``flow_id``
    (f"{dataset_code}-{original_csv_row_number}", 0-indexed) plus the same
    row number as an int ``source_row`` column -- pandas' chunked reader
    index is continuous across chunks within one ``read_csv`` call, so this
    matches the row's original line number.

    Returns (all_attacks_df, {dataset_code: benign_pool_df}).
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    chunk_size = cfg["sampling"]["chunk_size"]
    attack_quota: dict[tuple[str, str], int] = quotas["attack_quota"]
    benign_quota: dict[str, int] = quotas["benign_quota"]

    chosen_attack = {
        key: _choose_ordinals(attack_counts.get(key, 0), want, rng)
        for key, want in attack_quota.items()
    }
    chosen_benign = {
        code: _choose_ordinals(int(benign_counts.get(code, 0)), want, rng)
        for code, want in benign_quota.items()
    }
    seen_attack: Counter = Counter()
    seen_benign: Counter = Counter()

    attack_parts: list[pd.DataFrame] = []
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
                        continue  # no benign budget planned for this dataset
                    chosen = chosen_benign[code]
                    seen = seen_benign
                    seen_key: Any = code
                elif key in chosen_attack:
                    chosen = chosen_attack[key]
                    seen = seen_attack
                    seen_key = key
                else:
                    continue  # class not sanctioned for this dataset, or zero quota
                where = np.flatnonzero(canon == cls)
                ordinals = seen[seen_key] + np.arange(where.size, dtype=np.int64)
                seen[seen_key] += where.size
                keep = where[_ordinal_mask(chosen, ordinals)]
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
                    attack_parts.append(kept)

    _log_rss("pass2: before attacks concat")
    attacks = pd.concat(attack_parts, ignore_index=True) if attack_parts else pd.DataFrame()
    del attack_parts
    if not attacks.empty:
        attacks["task"] = attacks["canonical_label"].map(task_of).astype(int)
    _log_rss("pass2: after attacks concat")

    benign_pools: dict[str, pd.DataFrame] = {}
    for code, parts in benign_parts.items():
        benign_pools[code] = (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        )
    _log_rss("pass2: after benign_pools concat")
    return attacks, benign_pools


def _seed_from(rng: np.random.Generator) -> int:
    """Derive a fresh int seed from an rng (for pandas.sample reproducibility)."""
    return int(rng.integers(0, 2**31 - 1))


class BenignAllocator:
    """Hands each task a *disjoint* slice of every dataset's benign pool.

    The previous implementation called ``pool.sample()`` independently per
    task, so two tasks drawing from the same dataset (T3-T6 all draw from
    both ToN and CSE) could -- and measurably did -- land the same benign
    flow in one task's train split and another task's test split. Since the
    classifier head is shared across the whole 11-class label space, that is
    train/test contamination on the single largest class in the benchmark.

    Shuffling once per dataset and then walking a cursor makes the slices
    disjoint by construction, at no extra memory cost.
    """

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
    would overflow float32 at Step 2's feature cast.

    Checks every numeric column among ``original_cols`` (the original
    NetFlow schema, not the metadata columns added by this pipeline) --
    not just the specific column(s) observed to be corrupted in practice --
    since the same failure mode could in principle affect any of them.
    Returns (clean_frame, dropped_count).
    """
    numeric_cols = frame[original_cols].select_dtypes(include="number").columns
    values = frame[numeric_cols].to_numpy(dtype=np.float64)
    bad_row = ~np.isfinite(values) | (np.abs(values) > FLOAT32_MAX)
    bad = bad_row.any(axis=1)
    dropped = int(bad.sum())
    clean = frame.loc[~bad]
    # Equivalent to .reset_index(drop=True), but reassigning .index directly
    # replaces the Index object's metadata without pandas' reset_index
    # forcing a second full block-consolidation copy of clean's data on top
    # of the .loc[~bad] copy already taken above -- doubling peak memory for
    # no behavioral difference. Became load-bearing once benign_per_task
    # grew large enough (2026-08-11) that the biggest task's combined
    # attack+benign frame no longer fit through two full copies on this
    # machine's RAM (see pass2_sample's _log_rss for the matching fix there).
    clean.index = pd.RangeIndex(len(clean))
    return clean, dropped


def assemble_tasks(
    cfg: dict[str, Any],
    attacks: pd.DataFrame,
    benign_pools: dict[str, pd.DataFrame],
    original_cols: list[str],
    quotas: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build, split, and write one Parquet per task; return the manifest dict."""
    out_dir = Path(cfg["paths"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = cfg["split"]
    do_dedup = cfg["dedup"]
    out_cols = original_cols + META_COLS + ["split"]
    allocator = BenignAllocator(benign_pools, rng)

    manifest: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "sampling": cfg["sampling"],
        "split": ratios,
        "dedup": do_dedup,
        "quota_plan": {
            "attack_per_task": quotas["attack_per_task"],
            "benign_per_task": quotas["benign_per_task"],
            "attack_quota": {f"{c}|{d}": n for (c, d), n in sorted(quotas["attack_quota"].items())},
            "benign_alloc": {str(t): a for t, a in quotas["benign_alloc"].items()},
        },
        "tasks": {},
    }

    for task in sorted(TASK_THEMES):
        atk = attacks[attacks["task"] == task] if not attacks.empty else pd.DataFrame()
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

        manifest["tasks"][str(task)] = {
            "theme": TASK_THEMES[task],
            "datasets": list(TASK_DATASETS[task]),
            "n_rows": int(len(frame)),
            "duplicates_dropped": int(dropped),
            "corrupted_rows_dropped": int(corrupted_dropped),
            "class_counts": {k: int(v) for k, v in frame["canonical_label"].value_counts().items()},
            "split_counts": {k: int(v) for k, v in frame["split"].value_counts().items()},
            "source_counts": {k: int(v) for k, v in frame["source_dataset"].value_counts().items()},
            "file": path.name,
        }
        print(
            f"[assemble] task {task} ({TASK_THEMES[task]}): {len(frame):,} rows "
            f"-> {path.name} (dropped {dropped:,} dups, {corrupted_dropped:,} corrupted)",
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
        f"[pass1] done. (class, dataset) pairs: {len(attack_counts)}; "
        f"total available attack flows: {sum(attack_counts.values()):,}; "
        f"benign per dataset: {dict(benign_counts)}",
        flush=True,
    )
    # Sanity: every canonical attack class we expect actually appeared.
    missing = set(canonical_classes()) - {BENIGN} - {cls for cls, _ in attack_counts}
    if missing:
        print(f"[pass1] WARNING: expected classes absent from data: {sorted(missing)}")

    quotas = compute_quotas(
        attack_counts,
        benign_counts,
        cfg["sampling"]["attack_per_task"],
        cfg["sampling"]["benign_per_task"],
    )
    for task in sorted(TASK_THEMES):
        by_class = quotas["attack_per_task_by_class"][task]
        print(
            f"[quota] task {task} ({TASK_THEMES[task]}): "
            f"attack {sum(by_class.values()):,} {by_class}, "
            f"benign {sum(quotas['benign_alloc'][task].values()):,} "
            f"{quotas['benign_alloc'][task]}",
            flush=True,
        )

    attacks, benign_pools = pass2_sample(cfg, attack_counts, benign_counts, quotas, rng)
    print(
        f"[pass2] done. sampled attack flows: {len(attacks):,}; "
        f"benign pools: {{ {', '.join(f'{k}:{len(v):,}' for k, v in benign_pools.items())} }}",
        flush=True,
    )

    manifest = assemble_tasks(cfg, attacks, benign_pools, original_cols, quotas, rng)
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
