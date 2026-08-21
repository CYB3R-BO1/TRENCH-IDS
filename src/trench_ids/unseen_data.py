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


def _reservoir_update(
    reservoir: list[dict[str, Any]],
    seen_before: int,
    new_rows: pd.DataFrame,
    sample_cap: int,
    rng: np.random.Generator,
) -> int:
    """Algorithm R (reservoir sampling), applied in chunk-sized batches.

    `reservoir` is mutated in place and holds at most `sample_cap` rows,
    an exactly-uniform sample of every row passed to this function so far
    (across all calls, i.e. across the whole stream) -- this is what makes
    it safe to call once per CSV chunk without ever holding more than
    `sample_cap` rows in memory, and without biasing the sample toward
    whichever chunk happens to be seen last. `seen_before` is the count of
    rows passed to this function prior to this call; returns the updated
    total (`seen_before + len(new_rows)`).

    Only converts to Python dicts the rows that actually win a reservoir
    slot (`.to_dict("records")` on the whole chunk would be a
    memory/throughput regression on exactly the high-volume classes this
    is meant to bound -- expected winners across a full BoT-IoT-sized
    stream is O(sample_cap * log(n / sample_cap)), not O(n)).
    """
    seen = seen_before
    take = max(0, min(sample_cap - len(reservoir), len(new_rows)))

    # Fill phase: reservoir isn't full yet, so new rows go straight in.
    if take:
        fill_part = new_rows.iloc[:take]
        reservoir.extend(fill_part.to_dict("records"))
        seen += take

    # Replacement phase: for each remaining row, its 0-indexed global
    # position is `seen` (before incrementing). Draw j uniformly from
    # [0, seen] inclusive; replace reservoir[j] iff j < sample_cap. This
    # is exactly Algorithm R, vectorized in the random-draw step (the draw
    # for a row depends only on its position, not on any other row's
    # outcome, so batching the draws is safe) and materializes dicts only
    # for rows that win a slot.
    remaining = new_rows.iloc[take:]
    r = len(remaining)
    if r:
        positions = seen + np.arange(r)
        js = rng.integers(0, positions + 1)  # high is exclusive -> [0, position]
        winners = np.nonzero(js < sample_cap)[0]  # ascending order -> later row wins ties
        if len(winners):
            winner_js = js[winners]
            winner_records = remaining.iloc[winners].to_dict("records")
            for j, rec in zip(winner_js.tolist(), winner_records, strict=True):
                reservoir[j] = rec
        seen += r

    return seen


def _integral_target_dtypes(frame: pd.DataFrame) -> pd.Series:
    """``frame.dtypes``, with any float64 column downcast to int64 in the
    *reported* target if every value in ``frame`` is integral.

    Guards a specific failure this module's dtype-restoration mechanism
    otherwise has: ``reservoir_dtypes`` (below) captures its target from the
    *first* clean chunk, but pandas infers a whole chunk's dtype before
    ``_drop_corrupted_rows`` runs -- so if that first chunk had even one
    blank/NaN value anywhere in an otherwise-integer column (e.g.
    PROTOCOL), the whole column parses as float64 for that chunk, and stays
    float64 for the surviving rows even after the NaN-carrying row is
    dropped. Capturing that as the target makes ``_restore_dtypes`` a no-op
    -- it "restores" the frame to the same wrong dtype it already has,
    leaving e.g. PROTOCOL as ``6.0`` instead of ``6``, which then fails
    ``unseen_graphs.drop_vocab_gaps``'s ``str(int)`` vocab lookup and
    silently drops the whole class as a spurious "vocab gap". Recovering
    the column's true integer-ness here, from the data that actually
    survived filtering rather than from the chunk's raw parse, closes that
    gap regardless of which chunk happens to be "first".
    """
    dtypes = frame.dtypes.copy()
    for col in frame.columns:
        if dtypes[col] != np.float64:
            continue
        values = frame[col]
        if values.notna().all() and (values % 1 == 0).all():
            dtypes[col] = np.dtype("int64")
    return dtypes


def _restore_dtypes(frame: pd.DataFrame, target_dtypes: pd.Series) -> pd.DataFrame:
    """Best-effort per-column re-cast to `target_dtypes` (a chunk's original
    dtypes, captured before the dict round-trip in reservoir sampling).

    Per-column and non-fatal: a column that can't be safely cast (e.g. the
    winning reservoir rows genuinely span incompatible dtypes across
    chunks -- possible on a 45-column, many-chunk real CSV even though it
    doesn't happen in this module's own test fixtures) is left as pandas
    inferred it rather than raising and losing an otherwise-complete,
    multi-minute extraction. Works on a copy -- callers pass filtered
    slices, and assigning into a view would only raise SettingWithCopy
    warnings (or silently not persist).
    """
    frame = frame.copy()
    for col, dtype in target_dtypes.items():
        if frame[col].dtype == dtype:
            continue
        try:
            frame[col] = frame[col].astype(dtype)
        except (ValueError, TypeError):
            pass
    return frame


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
    Step 1), independently per chunk -- the check is row-local (no
    cross-row statistics), so this is equivalent to applying it once over
    the full concatenated set of matching rows. If more than `sample_cap`
    matching (and clean) rows exist, draws a seeded uniform sample of
    exactly `sample_cap` rows via single-pass reservoir sampling (Algorithm
    R -- see _reservoir_update) -- for high-volume classes like BoT-IoT's
    DDoS/DoS (18.3M/16.7M matching rows available) this bounds memory to
    `sample_cap` rows without ever concatenating the full matching set, and
    without biasing the sample toward rows seen later in the stream.
    Otherwise keeps every matching row. Returns (frame, manifest_entry) --
    manifest_entry has no "evaluation_type"/"used_in_training"/"output_path"
    keys yet; run() adds those since they're a property of the config
    entry, not of the extraction itself.
    """
    csv_path = _dataset_csv(raw_dir, dataset_dir)
    original_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    all_cols = original_cols + ["flow_id", "source_dataset", "canonical_label"]

    parts: list[pd.DataFrame] = []  # used only when sample_cap is None
    reservoir: list[dict[str, Any]] = []  # used only when sample_cap is set
    rows_found_total = 0
    corrupted_dropped_total = 0
    clean_seen_total = 0
    # Dtypes of the first clean matched chunk. Reservoir rows pass through a
    # Python-dict round trip (DataFrame.to_dict("records") / from_records),
    # which can silently up-cast columns (e.g. an int PROTOCOL column
    # becoming float64 if any dict along the way had a missing/NaN key);
    # the sample_cap=None concat path has the mirror-image problem: pandas
    # infers a whole chunk's dtype before _drop_corrupted_rows runs, so one
    # NaN in an otherwise-integer column leaves the surviving rows float64
    # ("6.0" not "6") in the concatenated parquet -- where it fails
    # unseen_graphs.drop_vocab_gaps's str(int) vocab lookup and silently
    # drops the whole class as a spurious "vocab gap". Restoring the
    # original dtypes in both paths keeps downstream consumers working.
    # Captured via _integral_target_dtypes, not clean.dtypes directly: the
    # *chunk* this "first clean" frame came from may itself have had an
    # unrelated NaN in the same column (later dropped by
    # _drop_corrupted_rows), which pandas would already have coerced to
    # float64 for the whole chunk -- see _integral_target_dtypes's
    # docstring for why capturing that coercion as the target makes the
    # restoration below a no-op.
    target_dtypes: pd.Series | None = None

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
        rows_found_total += len(matched)

        clean, corrupted_dropped = _drop_corrupted_rows(matched, original_cols)
        corrupted_dropped_total += corrupted_dropped
        if clean.empty:
            continue

        if target_dtypes is None:
            target_dtypes = _integral_target_dtypes(clean)

        if sample_cap is None:
            parts.append(_restore_dtypes(clean, target_dtypes))
        else:
            clean_seen_total = _reservoir_update(
                reservoir, clean_seen_total, clean, sample_cap, rng
            )

    rows_found = rows_found_total
    if sample_cap is None:
        frame = (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=all_cols)
        )
        sampled = False
    else:
        if reservoir:
            frame = pd.DataFrame.from_records(reservoir, columns=all_cols)
            frame = _restore_dtypes(frame, target_dtypes)
        else:
            frame = pd.DataFrame(columns=all_cols)
        sampled = clean_seen_total > sample_cap

    manifest_entry = {
        "canonical_label": canonical_label,
        "dataset": dataset_dir,
        "dataset_code": dataset_code,
        "rows_found": rows_found,
        "corrupted_rows_dropped": corrupted_dropped_total,
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
