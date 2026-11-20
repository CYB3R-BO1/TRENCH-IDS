#!/usr/bin/env python3
"""Step 1 -- Attack-Class Similarity Analysis for NetFlow-v3 datasets.

Supports NF-UNSW-NB15-v3 and NF-ToN-IoT-v3 (and any NetFlow-v3 CSV with the same
schema). Attack classes are DISCOVERED from the data's ``Attack`` column; no labels
are hardcoded.

This is *evidence*, not a task decision. All similarity metrics use the TRAINING
partition of one global chronological split (all-flow timestamp quantiles), so the
test period cannot influence task-design evidence. Invalid values are excluded
rather than zero-filled, and identifier-like codes are never treated as magnitudes.

Usage
-----
    python attack_class_similarity.py                    # default dataset (unsw)
    python attack_class_similarity.py --dataset ton
    python attack_class_similarity.py --data <csv> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

import feature_schema as fs  # noqa: E402
import windows as winmod  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BENIGN = "Benign"
DATASETS = {
    "unsw": ("v3/UNSW/data/NF-UNSW-NB15-v3.csv", "NF-UNSW-NB15-v3"),
    "ton": ("v3/ToN-IoT/NF-ToN-IoT-v3/data/NF-ToN-IoT-v3.csv", "NF-ToN-IoT-v3"),
}
DEFAULT_DATASET = "unsw"

IDENTITY_META = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    "PROTOCOL", "L7_PROTO",
]
LABEL_COLS = ["Label", "Attack"]
HOST_SRC = "IPV4_SRC_ADDR"
HOST_DST = "IPV4_DST_ADDR"

MAGNITUDE_FEATURES = fs.MAGNITUDE_FEATURES
DIST_FEATURES = fs.DIST_FEATURES
CATEGORICAL_FIELDS = fs.CATEGORICAL_FEATURES
BITMASK_FEATURES = fs.BITMASK_FEATURES

FLAG_MASKS = BITMASK_FEATURES
COSINE_EXCLUDE = frozenset(FLAG_MASKS)
FLAG_BITS = [("FIN", 0x01), ("SYN", 0x02), ("RST", 0x04), ("PSH", 0x08),
             ("ACK", 0x10), ("URG", 0x20), ("ECE", 0x40), ("CWR", 0x80)]
FLAG_BIT_NAMES = [f"{m}:{b}" for m in FLAG_MASKS for b, _ in FLAG_BITS]
FLAG_KIND_NAMES = ["valid", "nan", "inf", "negative", "fractional", "unsupported"]
FLAG_MAX_VALUE = 0xFF

LABEL_DOMAIN = (0, 1)
LABEL_ATTACK = 1
# Largest integer exactly representable in float64 (timestamps are parsed as float).
INT64_SAFE = 2 ** 53

CATEGORICAL_RANGES = {
    "PROTOCOL": (0, 255),
    "L7_PROTO": (0, 65535),
    "L4_SRC_PORT": (0, 65535),
    "L4_DST_PORT": (0, 65535),
    "TCP_FLAGS": (0, FLAG_MAX_VALUE),
    # Protocol-defined codes kept as categories (not discarded as identifiers).
    "ICMP_TYPE": (0, 65535),          # ICMP type*256 + code
    "ICMP_IPV4_TYPE": (0, 255),
    "DNS_QUERY_TYPE": (0, 65535),
    "FTP_COMMAND_RET_CODE": (0, 999),
}
CATEGORICAL_KIND_NAMES = ["valid", "nan", "inf", "fractional", "out_of_range"]

SPLIT_NAMES = ("train", "val", "test")

# Identity-free relation-profile blocks (source/destination roles preserved).
RELATION_BLOCKS = [
    "source_host_degree",   # flows per source host
    "dest_host_degree",     # flows per destination host
    "host_pair_degree",     # flows per (source,destination) host pair
    "flow_to_src_port",
    "flow_to_dst_port",
    "flow_to_protocol",
    "flow_to_service",
]


# --------------------------------------------------------------------------- #
# Path helpers
# --------------------------------------------------------------------------- #
def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / "v3").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    raise SystemExit(f"could not locate a repo root containing 'v3' above {start}")


STEP_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def _reservoir_update(res: np.ndarray, seen: int, rows: np.ndarray, rng, cap: int) -> int:
    n = rows.shape[0]
    fill_end = min(seen + n, cap)
    if seen < fill_end:
        res[seen:fill_end] = rows[: fill_end - seen]
    start = max(seen, cap)
    if seen + n > start:
        positions = np.arange(start, seen + n)
        probs = cap / (positions + 1.0)
        do = rng.random(len(positions)) < probs
        if do.any():
            local = positions[do] - seen
            slots = rng.integers(0, cap, size=int(do.sum()))
            res[slots] = rows[local]
    return seen + n


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine with explicit zero-vector policy: two zeros = 1, zero vs non-zero = 0."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    nz = norms > 0
    unit = np.zeros_like(vectors)
    unit[nz] = vectors[nz] / norms[nz, None]
    sim = unit @ unit.T
    zero = ~nz
    if zero.any():
        sim[np.ix_(zero, zero)] = 1.0
    np.fill_diagonal(sim, np.where(zero, 1.0, np.clip(np.diag(sim), -1.0, 1.0)))
    return np.clip(sim, -1.0, 1.0)


def defined_cosine(z: np.ndarray, labels: list) -> tuple[np.ndarray, list]:
    """Cosine over finite (jointly valid) magnitude features only.

    Pairs with no jointly-valid feature are undefined and raise (never NaN-filled).
    Zero standardised profiles use the zero-vector policy and are reported.
    """
    M = z.shape[0]
    mat = np.ones((M, M))
    zero_pairs = []
    for a in range(M):
        for b in range(a + 1, M):
            mask = np.isfinite(z[a]) & np.isfinite(z[b])
            if not mask.any():
                raise ValueError("no jointly valid magnitude features for classes "
                                 f"{labels[a]!r}, {labels[b]!r}")
            x, y = z[a][mask], z[b][mask]
            nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
            if nx == 0.0 and ny == 0.0:
                sim = 1.0
                zero_pairs.append((labels[a], labels[b]))
            elif nx == 0.0 or ny == 0.0:
                sim = 0.0
                zero_pairs.append((labels[a], labels[b]))
            else:
                sim = float(np.dot(x, y) / (nx * ny))
            mat[a, b] = mat[b, a] = sim
    return np.clip(mat, -1.0, 1.0), zero_pairs


def _prob_vector(counter: Counter, universe: list) -> np.ndarray:
    total = sum(counter.values())
    if total == 0:
        return np.zeros(len(universe) + 1)
    vec = np.array([counter.get(c, 0) for c in universe], dtype=np.float64)
    return np.concatenate([vec, [total - vec.sum()]]) / total


def _js(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(a, b, base=2))


def _merge_intervals(intervals: list) -> list:
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def _count_in_intervals(sorted_vals: np.ndarray, intervals: list) -> int:
    total = 0
    for lo, hi in intervals:
        total += int(np.searchsorted(sorted_vals, hi, "right")
                     - np.searchsorted(sorted_vals, lo, "left"))
    return total


def sorted_share_distribution(counter: Counter, length: int) -> np.ndarray:
    vals = np.sort(np.asarray(list(counter.values()), dtype=float))[::-1]
    v = np.zeros(length)
    v[: vals.size] = vals
    s = v.sum()
    return v / s if s else v


def episode_intervals(ts_sorted, gap_ms, margin_ms, clip_lo=None, clip_hi=None) -> list:
    if ts_sorted.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(ts_sorted) > gap_ms) + 1
    eps = np.split(ts_sorted, cuts)
    ivs = [(int(e[0]) - margin_ms, int(e[-1]) + margin_ms) for e in eps if e.size]
    if clip_lo is not None or clip_hi is not None:
        lo_c = -np.inf if clip_lo is None else clip_lo
        hi_c = np.inf if clip_hi is None else clip_hi
        ivs = [(max(lo, lo_c), min(hi, hi_c)) for lo, hi in ivs]
        ivs = [(lo, hi) for lo, hi in ivs if lo <= hi]
    return _merge_intervals(ivs)


def sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_csv(path: Path, cfg: dict, **kwargs):
    """Yield CSV chunks, honouring ``--max-rows`` in EVERY pass."""
    limit = cfg.get("max_rows")
    seen = 0
    for chunk in pd.read_csv(path, chunksize=cfg["chunk"], **kwargs):
        if limit is not None and seen + len(chunk) > limit:
            chunk = chunk.iloc[: max(limit - seen, 0)]
        if len(chunk):
            yield chunk
        seen += len(chunk)
        if limit is not None and seen >= limit:
            break


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def validate_labels(labels, attacks):
    labels = np.asarray(labels)
    attacks = np.asarray(attacks)
    counts = Counter(labels.tolist())
    bad = int(np.count_nonzero(~np.isin(labels, LABEL_DOMAIN)))
    mismatch = int(np.count_nonzero((labels == LABEL_ATTACK) != (attacks != BENIGN)))
    return counts, bad, mismatch


def validate_timestamp_column(values, name: str) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    integral = finite & (v == np.floor(v))
    safe = integral & (v >= -INT64_SAFE) & (v <= INT64_SAFE)
    bad = ~safe
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} value(s) are not finite, integral, "
                         f"and within +/-{INT64_SAFE:.0e}")
    return v.astype(np.int64)


def validate_timestamp_order(start, end) -> None:
    bad = np.asarray(end) < np.asarray(start)
    if bad.any():
        raise ValueError("FLOW_END_MILLISECONDS < FLOW_START_MILLISECONDS in "
                         f"{int(bad.sum())} row(s)")


def invalid_identity_mask(values) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=object))
    return (s.isna() | s.astype("string").str.strip().eq("").fillna(True)).to_numpy()


def continuous_invalid_masks(x):
    nan = np.isnan(x)
    inf = np.isinf(x)
    neg = np.isfinite(x) & (x < 0.0)
    return nan, inf, neg


def categorical_value_report(field: str, values):
    lo, hi = CATEGORICAL_RANGES[field]
    v = np.asarray(values, dtype=np.float64)
    nan = np.isnan(v)
    inf = np.isinf(v)
    finite = ~(nan | inf)
    integral = finite & (v == np.floor(v))
    fractional = finite & ~integral
    in_range = integral & (v >= lo) & (v <= hi)
    out_of_range = integral & ~in_range
    kinds = np.zeros(v.shape, dtype=np.int8)
    kinds[nan] = 1
    kinds[inf] = 2
    kinds[fractional] = 3
    kinds[out_of_range] = 4
    codes = np.zeros(v.shape, dtype=np.int64)
    codes[in_range] = v[in_range].astype(np.int64)
    return in_range, codes, kinds


def categorical_field_codes(field: str, values):
    if field in FLAG_MASKS:
        valid, codes, _ = flag_value_report(values)
        return valid, codes, None
    return categorical_value_report(field, values)


def flag_value_report(values):
    v = np.asarray(values, dtype=np.float64)
    nan = np.isnan(v)
    inf = np.isinf(v)
    finite = ~(nan | inf)
    neg = finite & (v < 0)
    nonneg = finite & (v >= 0)
    fractional = nonneg & (v != np.floor(v))
    integer_nonneg = nonneg & (v == np.floor(v))
    unsupported = integer_nonneg & (v > FLAG_MAX_VALUE)
    valid = integer_nonneg & (v <= FLAG_MAX_VALUE)
    kinds = np.zeros(v.shape, dtype=np.int8)
    kinds[nan] = 1
    kinds[inf] = 2
    kinds[neg] = 3
    kinds[fractional] = 4
    kinds[unsupported] = 5
    codes = np.zeros(v.shape, dtype=np.int64)
    codes[valid] = v[valid].astype(np.int64)
    return valid, codes, kinds


def flag_bit_prevalence(flagbit, valid_counts):
    n_bits = len(FLAG_BITS)
    den = np.repeat(valid_counts.astype(np.float64), n_bits, axis=1)
    prev = np.divide(flagbit.astype(np.float64), den, out=np.zeros_like(den), where=den > 0)
    return prev, den


def flag_support_gaps(valid_counts, class_names, mask_names) -> list:
    return [(class_names[ci], mask_names[mi])
            for ci in range(len(class_names))
            for mi in range(len(mask_names))
            if int(valid_counts[ci, mi]) == 0]


def validate_cfg(cfg: dict) -> None:
    numeric_lists = [("--split", list(cfg["split"])),
                     ("--window-sweep-ms", cfg["window_sweep_ms"]),
                     ("--sensitivity-gaps-min", cfg["sensitivity_gaps"]),
                     ("--sensitivity-margins-min", cfg["sensitivity_margins"]),
                     ("--margin-min", [cfg["margin_min"]]),
                     ("--episode-gap-min", [cfg["episode_gap_min"]])]
    for flag, values in numeric_lists:
        if not all(np.isfinite(float(x)) for x in values):
            raise SystemExit(f"invalid {flag} {values}: all values must be finite")
    s70, s85 = cfg["split"]
    if not (0.0 < s70 < s85 < 1.0):
        raise SystemExit(f"invalid --split {cfg['split']}: need 0 < train < val < 1")
    for key in ("window_ms", "reservoir_cap", "chunk", "split_floor", "min_graph_flows"):
        if cfg[key] <= 0:
            raise SystemExit(f"invalid --{key.replace('_', '-')} = {cfg[key]}: must be > 0")
    if cfg["min_feature_samples"] <= 0:
        raise SystemExit("invalid --min-feature-samples: must be > 0")
    if cfg["cat_topk"] < 0:
        raise SystemExit("invalid --cat-topk: must be >= 0 (0 = full support)")
    if cfg["margin_min"] < 0:
        raise SystemExit("invalid --margin-min: must be >= 0")
    if cfg["episode_gap_min"] <= 0:
        raise SystemExit("invalid --episode-gap-min: must be > 0")
    if any(w <= 0 for w in cfg["window_sweep_ms"]):
        raise SystemExit("invalid --window-sweep-ms: all must be > 0")
    if any(g <= 0 for g in cfg["sensitivity_gaps"]):
        raise SystemExit("invalid --sensitivity-gaps-min: all must be > 0")
    if any(m < 0 for m in cfg["sensitivity_margins"]):
        raise SystemExit("invalid --sensitivity-margins-min: must be >= 0")
    if cfg["seed"] < 0:
        raise SystemExit(f"invalid --seed = {cfg['seed']}: must be >= 0")


def aggregate_pairwise_js(items: list, M: int, labels=None):
    num = np.zeros((M, M))
    den = np.zeros((M, M), dtype=np.int64)
    per_item = {}
    for name, vecs, defined in items:
        mat = np.full((M, M), np.nan)
        for a in range(M):
            if not defined[a]:
                continue
            for b in range(a + 1, M):
                if not defined[b]:
                    continue
                d = _js(vecs[a], vecs[b])
                mat[a, b] = mat[b, a] = d
                num[a, b] += d
                num[b, a] += d
                den[a, b] += 1
                den[b, a] += 1
        per_item[name] = mat
    missing = []
    for i in range(M):
        for j in range(i + 1, M):
            if den[i, j] == 0:
                missing.append((labels[i] if labels else i, labels[j] if labels else j))
    if missing:
        raise ValueError(f"no categorical item defined for both classes in pair(s): {missing}")
    agg = np.zeros((M, M))
    mask = den > 0
    agg[mask] = num[mask] / den[mask]
    return agg, per_item, den


# --------------------------------------------------------------------------- #
# Dataset discovery + global split
# --------------------------------------------------------------------------- #
def discover_attack_classes(data_path: Path, cfg: dict) -> list[str]:
    counts: Counter = Counter()
    for chunk in _iter_csv(data_path, cfg, usecols=["Attack"]):
        counts.update(chunk["Attack"].value_counts().to_dict())
    attacks = sorted(k for k in counts if k != BENIGN)
    if not attacks:
        raise ValueError("no attack classes found: the Attack column contains only 'Benign'")
    return attacks


def compute_global_split_cuts(data_path: Path, cfg: dict, window_ms: int) -> tuple[int, int]:
    """Global chronological cuts from ALL rows' FLOW_START, aligned to window boundaries.

    Aligning to ``window_ms`` guarantees a physical graph window belongs to exactly
    one split, so train/val/test window mixing cannot leak across partitions.
    """
    s70, s85 = cfg["split"]
    parts = []
    for chunk in _iter_csv(data_path, cfg, usecols=["FLOW_START_MILLISECONDS"]):
        parts.append(validate_timestamp_column(
            chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
            "FLOW_START_MILLISECONDS"))
    ts = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if ts.size == 0:
        raise ValueError("no timestamps found")
    q = np.quantile(ts, [s70, s85])
    t70 = int(q[0] // window_ms * window_ms)
    t85 = int(q[1] // window_ms * window_ms)
    if t70 >= t85:
        raise ValueError(f"aligned split cuts are empty: t70={t70}, t85={t85}")
    return t70, t85


def load_taxonomy(cfg: dict) -> dict:
    taxo = {BENIGN: ("Benign", "Normal background traffic (shared negative class).")}
    path = cfg.get("taxonomy_csv")
    if path:
        df = pd.read_csv(path)
        col = "raw_label" if "raw_label" in df.columns else "class"
        for _, row in df.iterrows():
            family = row["family"] if "family" in df.columns and pd.notna(row["family"]) else str(row[col])
            desc = row["description"] if "description" in df.columns and pd.notna(row["description"]) else ""
            taxo[str(row[col])] = (str(family), str(desc))
    return taxo


def class_taxonomy(name: str, taxo: dict) -> tuple[str, str]:
    return taxo.get(name, (name, ""))


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract(data_path: Path, cfg: dict, attack_classes: list[str], cuts: tuple[int, int]) -> dict:
    all_cols = pd.read_csv(data_path, nrows=0).columns.tolist()
    exclude = set(cfg.get("exclude_features") or [])
    unknown = exclude - set(MAGNITUDE_FEATURES) - set(DIST_FEATURES)
    if unknown:
        raise ValueError(f"unknown --exclude-features: {sorted(unknown)}")
    features = [f for f in MAGNITUDE_FEATURES if f in all_cols and f not in exclude]
    dist_features = [f for f in DIST_FEATURES if f not in exclude]
    required = set(features + FLAG_MASKS + CATEGORICAL_FIELDS +
                   [HOST_SRC, HOST_DST, "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
                    "Label", "Attack"])
    missing = sorted(required - set(all_cols))
    if missing:
        raise ValueError(f"expected columns missing from CSV: {missing}")
    F = len(features)
    all_classes = [BENIGN] + list(attack_classes)
    code_of = {name: i for i, name in enumerate(all_classes)}
    K = len(all_classes)
    t70, t85 = cuts
    cap = cfg["reservoir_cap"]
    rng = np.random.default_rng(cfg["seed"])
    print(f"[extract] {len(all_cols)} columns -> {F} magnitude features; "
          f"{len(all_classes)} classes; cuts=({t70}, {t85})")

    # Train-only similarity accumulators.
    g_valid = np.zeros(F, dtype=np.int64)
    g_sum = np.zeros(F)
    g_sumsq = np.zeros(F)
    c_valid = np.zeros((K, F), dtype=np.int64)
    c_sum = np.zeros((K, F))
    c_sumsq = np.zeros((K, F))
    res = np.full((K, cap, F), np.nan, dtype=np.float32)
    seen = np.zeros(K, dtype=np.int64)
    have_res = np.array([n != BENIGN for n in all_classes])
    src_ct = [Counter() for _ in range(K)]
    dst_ct = [Counter() for _ in range(K)]
    pair_ct = [Counter() for _ in range(K)]
    cat_ct = [[Counter() for _ in CATEGORICAL_FIELDS] for _ in range(K)]
    flagbit = np.zeros((K, len(FLAG_BIT_NAMES)), dtype=np.int64)
    c_flag_valid = np.zeros((K, len(FLAG_MASKS)), dtype=np.int64)

    g_nan = np.zeros(F, dtype=np.int64)
    g_inf = np.zeros(F, dtype=np.int64)
    g_neg = np.zeros(F, dtype=np.int64)
    c_nan = np.zeros((K, F), dtype=np.int64)
    c_inf = np.zeros((K, F), dtype=np.int64)
    c_neg = np.zeros((K, F), dtype=np.int64)
    n_cat = len(CATEGORICAL_FIELDS)
    n_flag = len(FLAG_MASKS)
    g_cat_kind = np.zeros((n_cat, len(CATEGORICAL_KIND_NAMES)), dtype=np.int64)
    c_cat_kind = np.zeros((K, n_cat, len(CATEGORICAL_KIND_NAMES)), dtype=np.int64)
    g_flag_kind = np.zeros((n_flag, len(FLAG_KIND_NAMES)), dtype=np.int64)
    c_flag_kind = np.zeros((K, n_flag, len(FLAG_KIND_NAMES)), dtype=np.int64)

    # All-split feasibility accumulators.
    ccount_split = np.zeros((K, 3), dtype=np.int64)
    cat_valid_split = np.zeros((K, 3, n_cat), dtype=np.int64)
    flag_valid_split = np.zeros((K, 3, n_flag), dtype=np.int64)
    src_ct_split = [[Counter() for _ in range(3)] for _ in range(K)]
    dst_ct_split = [[Counter() for _ in range(3)] for _ in range(K)]
    ts_parts = [[] for _ in range(K)]

    label_counts = Counter()
    attack_counts = Counter()
    label_mismatch = 0
    n_rows = 0

    for chunk in _iter_csv(data_path, cfg):
        n_rows += len(chunk)
        codes = chunk["Attack"].map(code_of)
        if codes.isna().any():
            unknown = sorted(set(chunk.loc[codes.isna(), "Attack"].unique()))
            raise ValueError(f"unmapped Attack label(s): {unknown}")
        codes = codes.to_numpy(dtype=np.int16)

        counts, bad, mismatch = validate_labels(chunk["Label"].to_numpy(),
                                                chunk["Attack"].to_numpy())
        label_counts.update(counts)
        if bad:
            raise ValueError(f"unexpected Label value(s); allowed {LABEL_DOMAIN}")
        label_mismatch += mismatch
        attack_counts.update(chunk["Attack"].value_counts().to_dict())

        t = validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                      "FLOW_START_MILLISECONDS")
        e = validate_timestamp_column(chunk["FLOW_END_MILLISECONDS"].to_numpy(dtype=np.float64),
                                      "FLOW_END_MILLISECONDS")
        validate_timestamp_order(t, e)
        src = chunk[HOST_SRC].to_numpy()
        dst = chunk[HOST_DST].to_numpy()
        src_bad = invalid_identity_mask(src)
        dst_bad = invalid_identity_mask(dst)
        if src_bad.any() or dst_bad.any():
            raise ValueError("invalid (null/empty/blank) host identity: "
                             f"src={int(src_bad.sum())} dst={int(dst_bad.sum())}")

        split = np.where(t < t70, 0, np.where(t < t85, 1, 2)).astype(np.int8)
        is_train = split == 0

        X = chunk[features].to_numpy(dtype=np.float64)
        nan_m, inf_m, neg_m = continuous_invalid_masks(X)
        g_nan += nan_m.sum(axis=0)
        g_inf += inf_m.sum(axis=0)
        g_neg += neg_m.sum(axis=0)
        valid = np.isfinite(X) & (X >= 0.0)
        safe = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        L = np.log1p(safe)
        L[~valid] = np.nan

        if is_train.any():
            Lt = L[is_train]
            g_valid += np.isfinite(Lt).sum(axis=0)
            g_sum += np.nansum(Lt, axis=0)
            g_sumsq += np.nansum(Lt * Lt, axis=0)

        cat_vals, cat_valid, cat_kinds = {}, {}, []
        for j, field in enumerate(CATEGORICAL_FIELDS):
            v, codes_col, kinds = categorical_field_codes(field, chunk[field].to_numpy(dtype=np.float64))
            cat_vals[field] = codes_col
            cat_valid[field] = v
            cat_kinds.append(kinds)
            if kinds is not None:
                g_cat_kind[j] += np.bincount(kinds, minlength=len(CATEGORICAL_KIND_NAMES))[:len(CATEGORICAL_KIND_NAMES)]

        bitmat = np.zeros((len(chunk), len(FLAG_BIT_NAMES)), dtype=bool)
        flag_valid_masks, flag_kinds = [], []
        col = 0
        for k, mask in enumerate(FLAG_MASKS):
            v, codes_col, kinds = flag_value_report(chunk[mask].to_numpy(dtype=np.float64))
            flag_valid_masks.append(v)
            flag_kinds.append(kinds)
            g_flag_kind[k] += np.bincount(kinds, minlength=len(FLAG_KIND_NAMES))[:len(FLAG_KIND_NAMES)]
            for _, bit in FLAG_BITS:
                bitmat[:, col] = v & ((codes_col & bit) != 0)
                col += 1

        for c in np.unique(codes):
            idx = np.flatnonzero(codes == c)
            sp = split[idx]
            ccount_split[c] += np.bincount(sp, minlength=3)
            ts_parts[c].append(t[idx].copy())
            for s in range(3):
                sidx = idx[sp == s]
                if sidx.size:
                    for j, field in enumerate(CATEGORICAL_FIELDS):
                        cat_valid_split[c, s, j] += int(cat_valid[field][sidx].sum())
                    for k in range(n_flag):
                        flag_valid_split[c, s, k] += int(flag_valid_masks[k][sidx].sum())
                    src_ct_split[c][s].update(src[sidx].tolist())
                    dst_ct_split[c][s].update(dst[sidx].tolist())

            tr = idx[sp == 0]
            if tr.size:
                Ltr = L[tr]
                c_valid[c] += np.isfinite(Ltr).sum(axis=0)
                c_sum[c] += np.nansum(Ltr, axis=0)
                c_sumsq[c] += np.nansum(Ltr * Ltr, axis=0)
                c_nan[c] += nan_m[tr].sum(axis=0)
                c_inf[c] += inf_m[tr].sum(axis=0)
                c_neg[c] += neg_m[tr].sum(axis=0)
                src_ct[c].update(src[tr].tolist())
                dst_ct[c].update(dst[tr].tolist())
                pair_ct[c].update(zip(src[tr].tolist(), dst[tr].tolist()))
                for j, field in enumerate(CATEGORICAL_FIELDS):
                    vmask = cat_valid[field][tr]
                    cat_ct[c][j].update(cat_vals[field][tr][vmask].tolist())
                    kinds = cat_kinds[j]
                    if kinds is not None:
                        c_cat_kind[c, j] += np.bincount(kinds[tr], minlength=len(CATEGORICAL_KIND_NAMES))[:len(CATEGORICAL_KIND_NAMES)]
                for k in range(n_flag):
                    c_flag_valid[c, k] += int(flag_valid_masks[k][tr].sum())
                    c_flag_kind[c, k] += np.bincount(flag_kinds[k][tr], minlength=len(FLAG_KIND_NAMES))[:len(FLAG_KIND_NAMES)]
                flagbit[c] += bitmat[tr].sum(axis=0)
                seen[c] = _reservoir_update(res[c], int(seen[c]), np.asarray(Ltr, dtype=np.float32), rng, cap)

        if n_rows % 1_000_000 < cfg["chunk"]:
            print(f"  scanned {n_rows:,} rows ...")

    print(f"[extract] done: {n_rows:,} rows")
    return {
        "n_rows": n_rows, "dataset_name": cfg.get("dataset_name", ""),
        "all_classes": all_classes, "matrix_classes": list(attack_classes),
        "magnitude_features": features, "dist_features": dist_features,
        "F": F, "K": K, "cuts": (t70, t85),
        "g_valid": g_valid, "g_sum": g_sum, "g_sumsq": g_sumsq,
        "c_valid": c_valid, "c_sum": c_sum, "c_sumsq": c_sumsq,
        "res": res, "seen": seen, "have_res": have_res,
        "g_nan": g_nan, "g_inf": g_inf, "g_neg": g_neg,
        "c_nan": c_nan, "c_inf": c_inf, "c_neg": c_neg,
        "g_cat_kind": g_cat_kind, "c_cat_kind": c_cat_kind,
        "g_flag_kind": g_flag_kind, "c_flag_kind": c_flag_kind,
        "src_ct": src_ct, "dst_ct": dst_ct, "pair_ct": pair_ct,
        "cat_ct": cat_ct, "flagbit": flagbit, "c_flag_valid": c_flag_valid,
        "ts_parts": ts_parts, "ccount_split": ccount_split,
        "cat_valid_split": cat_valid_split, "flag_valid_split": flag_valid_split,
        "src_ct_split": src_ct_split, "dst_ct_split": dst_ct_split,
        "label_counts": label_counts, "attack_counts": attack_counts,
        "label_mismatch": label_mismatch,
    }


# --------------------------------------------------------------------------- #
# Similarity matrices
# --------------------------------------------------------------------------- #
def build_matrices(d: dict, cfg: dict) -> dict:
    names = d["all_classes"]
    matrix_classes = d["matrix_classes"]
    K, F = d["K"], d["F"]
    M = len(matrix_classes)
    class_code = {name: c for c, name in enumerate(names)}
    m_idx = [names.index(c) for c in matrix_classes]

    gv = d["g_valid"]
    gmean = np.divide(d["g_sum"], gv, out=np.zeros(F), where=gv > 0)
    gvar = np.divide(d["g_sumsq"], gv, out=np.zeros(F), where=gv > 0) - gmean ** 2
    gstd = np.sqrt(np.clip(gvar, 0.0, None))
    gstd[gstd == 0] = 1.0

    cv = d["c_valid"]
    cmean = np.divide(d["c_sum"], cv, out=np.full((K, F), np.nan), where=cv > 0)
    zprofile = (cmean - gmean) / gstd

    # Consistent feature policy: a feature is used only if EVERY matrix class has
    # at least min_feature_samples valid training observations. This keeps the
    # main cosine comparison on one common feature set across all pairs.
    support = cv[m_idx]  # (M, F) valid counts
    consistent = [p for p in range(F) if int(support[:, p].min()) >= cfg["min_feature_samples"]]
    if len(consistent) < 2:
        raise ValueError("fewer than 2 magnitude features have consistent training support "
                         "for all classes")
    za = zprofile[np.ix_(m_idx, consistent)]
    cosine = cosine_matrix(za)
    zero_profile_classes = [matrix_classes[i] for i in range(M) if not np.any(za[i] != 0)]
    zero_pairs = [(matrix_classes[a], matrix_classes[b])
                  for a in range(M) for b in range(a + 1, M)
                  if (not np.any(za[a] != 0)) or (not np.any(za[b] != 0))]
    # Supplementary pair coverage: jointly finite magnitude features per pair.
    cosine_pair_coverage = np.zeros((M, M), dtype=np.int64)
    for a in range(M):
        for b in range(a + 1, M):
            shared = int((np.isfinite(zprofile[m_idx[a]]) &
                          np.isfinite(zprofile[m_idx[b]])).sum())
            cosine_pair_coverage[a, b] = cosine_pair_coverage[b, a] = shared
    cosine_feature_support = support.copy()

    # Wasserstein on STANDARDISED log values (train statistics), per feature.
    feat_pos = [d["magnitude_features"].index(f) for f in d.get("dist_features", DIST_FEATURES)]
    samples = {}
    for name in matrix_classes:
        ci = class_code[name]
        n = min(int(d["seen"][ci]), cfg["reservoir_cap"])
        col = d["res"][ci][:n]
        samples[name] = {
            p: (col[:, p][np.isfinite(col[:, p])] - gmean[p]) / gstd[p]
            for p in feat_pos
        }
    min_s = cfg["min_feature_samples"]
    wass_per_feature = {}
    wass_support = np.zeros((M, len(feat_pos)), dtype=np.int64)
    num = np.zeros((M, M))
    den = np.zeros((M, M), dtype=np.int64)
    for k, p in enumerate(feat_pos):
        fm = np.full((M, M), np.nan)
        for a in range(M):
            sa = samples[matrix_classes[a]][p]
            wass_support[a, k] = sa.size
            for b in range(a + 1, M):
                sb = samples[matrix_classes[b]][p]
                if sa.size >= min_s and sb.size >= min_s:
                    w = float(wasserstein_distance(sa, sb))
                    fm[a, b] = fm[b, a] = w
                    num[a, b] += w
                    num[b, a] += w
                    den[a, b] += 1
                    den[b, a] += 1
        wass_per_feature[d["magnitude_features"][p]] = fm
    missing = [(matrix_classes[i], matrix_classes[j]) for i in range(M)
               for j in range(i + 1, M) if den[i, j] == 0]
    if missing:
        raise ValueError(f"no feature had enough valid samples for pair(s): {missing}")
    wass = np.divide(num, den, out=np.zeros((M, M)), where=den > 0)

    cat_topk = cfg["cat_topk"]
    field_universe, compression = {}, {}
    for j, field in enumerate(CATEGORICAL_FIELDS):
        glob = Counter()
        for c in range(K):
            glob.update(d["cat_ct"][c][j])
        total = sum(glob.values())
        n_keep = None if cat_topk <= 0 else cat_topk
        universe = [c for c, _ in glob.most_common(n_keep)]
        field_universe[field] = universe
        kept = sum(v for _, v in glob.most_common(n_keep)) if total else 0
        compression[field] = {"n_categories": len(universe), "total": total,
                              "kept_mass": (kept / total if total else 1.0)}
    cat_items = []
    for j, field in enumerate(CATEGORICAL_FIELDS):
        vecs = [_prob_vector(d["cat_ct"][class_code[name]][j], field_universe[field])
                for name in matrix_classes]
        cat_items.append((field, vecs, [v.sum() > 0 for v in vecs]))
    js, js_per_field, js_defined = aggregate_pairwise_js(cat_items, M, matrix_classes)

    L_host = max(max(len(d["src_ct"][c]), len(d["dst_ct"][c])) for c in range(K))
    L_pair = max(len(d["pair_ct"][c]) for c in range(K))

    def _block_dist(ci: int, block: str) -> np.ndarray:
        if block == "source_host_degree":
            return sorted_share_distribution(d["src_ct"][ci], L_host)
        if block == "dest_host_degree":
            return sorted_share_distribution(d["dst_ct"][ci], L_host)
        if block == "host_pair_degree":
            return sorted_share_distribution(d["pair_ct"][ci], L_pair)
        if block == "flow_to_src_port":
            return _prob_vector(d["cat_ct"][ci][CATEGORICAL_FIELDS.index("L4_SRC_PORT")], field_universe["L4_SRC_PORT"])
        if block == "flow_to_dst_port":
            return _prob_vector(d["cat_ct"][ci][CATEGORICAL_FIELDS.index("L4_DST_PORT")], field_universe["L4_DST_PORT"])
        if block == "flow_to_protocol":
            return _prob_vector(d["cat_ct"][ci][CATEGORICAL_FIELDS.index("PROTOCOL")], field_universe["PROTOCOL"])
        if block == "flow_to_service":
            return _prob_vector(d["cat_ct"][ci][CATEGORICAL_FIELDS.index("L7_PROTO")], field_universe["L7_PROTO"])
        raise KeyError(block)

    rel_items = []
    for b in RELATION_BLOCKS:
        vecs = [_block_dist(class_code[name], b) for name in matrix_classes]
        rel_items.append((b, vecs, [v.sum() > 0 for v in vecs]))
    struct, block_mat, rel_defined = aggregate_pairwise_js(rel_items, M, matrix_classes)
    relation_similarity = np.clip(1.0 - struct, 0.0, 1.0)

    src_sets = {name: set(d["src_ct"][class_code[name]]) for name in matrix_classes}
    dst_sets = {name: set(d["dst_ct"][class_code[name]]) for name in matrix_classes}

    def _jaccard(a, b):
        u = a | b
        return len(a & b) / len(u) if u else 0.0

    host_overlap = np.zeros((M, M, 2))
    for a in range(M):
        for b in range(M):
            host_overlap[a, b, 0] = _jaccard(src_sets[matrix_classes[a]], src_sets[matrix_classes[b]])
            host_overlap[a, b, 1] = _jaccard(dst_sets[matrix_classes[a]], dst_sets[matrix_classes[b]])

    flag_valid_counts = d["c_flag_valid"]
    flag_prev, flag_den = flag_bit_prevalence(d["flagbit"], flag_valid_counts)
    flag_sim = cosine_matrix(flag_prev[m_idx])
    flag_total_valid = flag_valid_counts.sum(axis=1)
    flag_missing_classes = [names[c] for c in range(K) if flag_total_valid[c] == 0]
    flag_gaps = flag_support_gaps(flag_valid_counts, names, FLAG_MASKS)

    return {
        "gmean": gmean, "gstd": gstd,
        "cosine": cosine, "zero_cosine_pairs": zero_pairs,
        "cosine_pair_coverage": cosine_pair_coverage,
        "cosine_feature_support": cosine_feature_support,
        "cosine_consistent_features": consistent,
        "wasserstein": wass, "wasserstein_per_feature": wass_per_feature,
        "wasserstein_feature_support": wass_support,
        "wasserstein_pair_coverage": den,
        "jensen_shannon": js, "js_per_field": js_per_field, "js_defined_counts": js_defined,
        "relation_profile": relation_similarity, "relation_block_distance": block_mat,
        "relation_defined_counts": rel_defined,
        "host_overlap": host_overlap,
        "flag_bit_prevalence": flag_prev, "flag_bit_similarity": flag_sim,
        "flag_bit_denominator": flag_den, "flag_valid_counts": flag_valid_counts,
        "flag_missing_classes": flag_missing_classes, "flag_support_gaps": flag_gaps,
        "zero_profile_classes": zero_profile_classes,
        "field_universe": field_universe, "categorical_compression": compression,
    }


# --------------------------------------------------------------------------- #
# Feasibility
# --------------------------------------------------------------------------- #
def build_feasibility(d: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = d["all_classes"]
    t70, t85 = d["cuts"]
    floor = cfg["split_floor"]
    benign_ts = (np.sort(np.concatenate(d["ts_parts"][0])) if d["ts_parts"][0]
                 else np.array([], np.int64))
    benign_split = {
        "train": int((benign_ts < t70).sum()),
        "val": int(((benign_ts >= t70) & (benign_ts < t85)).sum()),
        "test": int((benign_ts >= t85).sum()),
    }
    rows, sweep_rows = [], []
    for c, name in enumerate(names):
        counts = d["ccount_split"][c]
        if counts.sum() == 0:
            continue
        ts = np.sort(np.concatenate(d["ts_parts"][c]))
        n = ts.size
        train, val, test = (int(counts[0]), int(counts[1]), int(counts[2]))
        attack_floor_ok = (min(train, val, test) >= floor) if name != BENIGN else None
        ben_ok = {s: benign_split[s] >= floor for s in SPLIT_NAMES}
        rows.append({
            "class": name, "is_attack": name != BENIGN, "flows": int(n),
            "train_flows": train, "val_flows": val, "test_flows": test,
            "min_split_flows": min(train, val, test),
            "attack_floor_ok": ("" if attack_floor_ok is None else bool(attack_floor_ok)),
            "benign_train": benign_split["train"], "benign_val": benign_split["val"],
            "benign_test": benign_split["test"],
            "benign_ok_train": bool(ben_ok["train"]), "benign_ok_val": bool(ben_ok["val"]),
            "benign_ok_test": bool(ben_ok["test"]),
            "unique_src_hosts_train": len(d["src_ct_split"][c][0]),
            "unique_src_hosts_val": len(d["src_ct_split"][c][1]),
            "unique_src_hosts_test": len(d["src_ct_split"][c][2]),
            "unique_dst_hosts_train": len(d["dst_ct_split"][c][0]),
            "unique_dst_hosts_val": len(d["dst_ct_split"][c][1]),
            "unique_dst_hosts_test": len(d["dst_ct_split"][c][2]),
            "first_timestamp": str(pd.to_datetime(int(ts[0]), unit="ms")),
            "last_timestamp": str(pd.to_datetime(int(ts[-1]), unit="ms")),
            "span_hours": round((int(ts[-1]) - int(ts[0])) / 3_600_000, 2),
            "split_basis": "global chronological all-flow quantiles (train-only metrics)",
            "feasibility_note": ("" if name == BENIGN or attack_floor_ok else
                                 f"attack min split {min(train, val, test)} < floor {floor}"),
        })
        for size in cfg["window_sweep_ms"]:
            w = ts // size
            _, cnt = np.unique(w, return_counts=True)
            sweep_rows.append({"class": name, "window_ms": size,
                               "nonempty_windows": int(np.unique(w).size),
                               "median_flows_per_window": int(np.median(cnt)),
                               "max_flows_per_window": int(cnt.max())})
    return pd.DataFrame(rows), pd.DataFrame(sweep_rows)


def build_margin_gap_sensitivity(d: dict, cfg: dict) -> pd.DataFrame:
    """Per-split contemporaneous benign under the GLOBAL split (not per-class cuts)."""
    t70, t85 = d["cuts"]
    benign_ts = (np.sort(np.concatenate(d["ts_parts"][0])) if d["ts_parts"][0]
                 else np.array([], np.int64))
    rows = []
    for c, name in enumerate(d["all_classes"]):
        if int(d["ccount_split"][c].sum()) == 0:
            continue
        ts = np.sort(np.concatenate(d["ts_parts"][c]))
        split_ts = {"train": ts[ts < t70], "val": ts[(ts >= t70) & (ts < t85)],
                    "test": ts[ts >= t85]}
        # Clip guard margins to the GLOBAL split boundaries, not the first/last
        # attack timestamp, so benign flows around episodes are counted correctly.
        bounds = {"train": (None, t70 - 1), "val": (t70, t85 - 1), "test": (t85, None)}
        for gap_min in cfg["sensitivity_gaps"]:
            for margin_min in cfg["sensitivity_margins"]:
                gap, margin = gap_min * 60_000, margin_min * 60_000
                if name == BENIGN:
                    bsplit = {s: int(sub.size) for s, sub in split_ts.items()}
                else:
                    bsplit = {}
                    for s, sub in split_ts.items():
                        if sub.size == 0:
                            bsplit[s] = 0
                        else:
                            lo, hi = bounds[s]
                            iv = episode_intervals(sub, gap, margin, clip_lo=lo, clip_hi=hi)
                            bsplit[s] = int(_count_in_intervals(benign_ts, iv))
                rows.append({"class": name, "episode_gap_min": gap_min, "margin_min": margin_min,
                             "benign_contemp_train": bsplit["train"],
                             "benign_contemp_val": bsplit["val"],
                             "benign_contemp_test": bsplit["test"]})
    return pd.DataFrame(rows)


def train_support_scope(d: dict, cfg: dict) -> tuple[list[str], pd.DataFrame]:
    """Split discovered classes by training support (all classes stay in feasibility)."""
    rows, supported = [], []
    threshold = max(int(cfg["split_floor"]), int(cfg["min_train_flows"]))
    for c, name in enumerate(d["all_classes"]):
        tr = int(d["ccount_split"][c, 0])
        if tr == 0:
            status = "zero_train_support"
        elif tr < threshold:
            status = "low_train_support"
        else:
            status = "train_supported"
        if name != BENIGN and tr >= threshold:
            supported.append(name)
        rows.append({"class": name, "train_flows": tr, "status": status})
    return supported, pd.DataFrame(rows)


def undefined_pairs(scope_df: pd.DataFrame, matrix_classes: list) -> pd.DataFrame:
    """Pairs excluded from the similarity matrices, with the reason."""
    supported = set(matrix_classes)
    status = {r["class"]: r["status"] for _, r in scope_df.iterrows()}
    excluded = [c for c, s in status.items() if c != BENIGN and c not in supported]
    rows = [{"class_a": a, "class_b": b, "defined": False,
             "reason": f"{b} has {status[b]}"}
            for a in matrix_classes for b in excluded]
    return pd.DataFrame(rows, columns=["class_a", "class_b", "defined", "reason"])


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def _save_matrix(mat, labels, csv_path, title, png_path, cmap, vmin=None, vmax=None):
    df = pd.DataFrame(np.asarray(mat, dtype=float), index=labels, columns=labels)
    df.to_csv(csv_path, float_format="%.6f")
    fig, ax = plt.subplots(figsize=(1.0 + 0.85 * len(labels), 0.9 + 0.78 * len(labels)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap=cmap, vmin=vmin, vmax=vmax, square=True,
                linewidths=0.5, ax=ax, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _longform(mat, labels, value_name):
    rows = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            val = mat[a, b]
            defined = bool(np.isfinite(val))
            rows.append({"class_a": labels[a], "class_b": labels[b],
                         value_name: (round(float(val), 6) if defined else np.nan),
                         "defined": defined})
    return pd.DataFrame(rows)


def _rankings(labels, values, name, ascending):
    pairs = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            pairs.append({"class_a": labels[a], "class_b": labels[b], "value": float(values[a, b])})
    df = pd.DataFrame(pairs)
    df["rank"] = df["value"].rank(ascending=ascending, method="min").astype(int)
    return df.sort_values("rank").rename(columns={"value": name})


def write_outputs(d, mats, feas, sweep, sens, window_feas, cfg, prov) -> dict:
    labels = d["matrix_classes"]
    all_classes = d["all_classes"]
    taxo = load_taxonomy(cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    benign_total = int(d["attack_counts"].get(BENIGN, 0))
    total_rows = d["n_rows"]

    with open(OUT_DIR / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
    with open(OUT_DIR / "feature_schema.json", "w", encoding="utf-8") as fh:
        json.dump({"roles": fs.feature_roles(), "magnitude_features": d["magnitude_features"],
                   "dist_features": d.get("dist_features", DIST_FEATURES), "identifier_features": fs.IDENTIFIER_FEATURES,
                   "bitmask_features": fs.BITMASK_FEATURES,
                   "categorical_fields": CATEGORICAL_FIELDS}, fh, indent=2)
    split_rows = {s: int(d["ccount_split"][:, i].sum()) for i, s in enumerate(SPLIT_NAMES)}
    split_manifest = {
        "cuts": {"t70": int(d["cuts"][0]), "t85": int(d["cuts"][1])},
        "cuts_aligned_to_window": True,
        "window_ms": cfg["window_ms"],
        "rows": split_rows,
        "percent": {s: (round(100 * split_rows[s] / total_rows, 3) if total_rows else 0.0)
                    for s in SPLIT_NAMES},
        "attack_rows": {s: int(d["ccount_split"][1:, i].sum()) for i, s in enumerate(SPLIT_NAMES)},
        "benign_rows": {s: int(d["ccount_split"][0, i]) for i, s in enumerate(SPLIT_NAMES)},
        "sha256": prov["data_sha256"],
    }
    with open(OUT_DIR / "split_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2)

    supported, scope_df = train_support_scope(d, cfg)
    scope_df.to_csv(OUT_DIR / "class_scope.csv", index=False)
    undefined_pairs(scope_df, labels).to_csv(OUT_DIR / "undefined_pairs.csv", index=False)

    rows = []
    for name in all_classes:
        cnt = int(d["attack_counts"].get(name, 0))
        fam, desc = class_taxonomy(name, taxo)
        rows.append({"class": name, "family": fam, "flows": cnt,
                     "share_of_file_pct": round(100 * cnt / total_rows, 4) if total_rows else 0.0,
                     "description": desc})
    counts_df = pd.DataFrame(rows)
    counts_df.to_csv(OUT_DIR / "attack_class_counts.csv", index=False)
    counts_df[["class", "family", "description"]].to_csv(OUT_DIR / "taxonomy.csv", index=False)

    _save_matrix(mats["cosine"], labels, OUT_DIR / "standardized_mean_profile_cosine.csv",
                 "Standardised mean-profile cosine (train-only)",
                 FIG_DIR / "standardized_mean_profile_cosine.png", "viridis", vmin=-1, vmax=1)
    _save_matrix(mats["wasserstein"], labels, OUT_DIR / "wasserstein_distance.csv",
                 "Wasserstein distance (valid samples, train-only)",
                 FIG_DIR / "wasserstein_distance.png", "magma_r")
    _save_matrix(mats["jensen_shannon"], labels, OUT_DIR / "jensen_shannon_distance.csv",
                 "Jensen-Shannon distance (train-only)",
                 FIG_DIR / "jensen_shannon_distance.png", "cividis_r")
    _save_matrix(mats["relation_profile"], labels, OUT_DIR / "relation_profile_similarity.csv",
                 "Relation-profile similarity (identity-free, train-only)",
                 FIG_DIR / "relation_profile_similarity.png", "viridis", vmin=0, vmax=1)
    _save_matrix(mats["host_overlap"][:, :, 0], labels, OUT_DIR / "host_identity_overlap_source.csv",
                 "Source-host identity overlap (Jaccard) - SEPARATE from structure",
                 FIG_DIR / "host_identity_overlap_source.png", "Blues", vmin=0, vmax=1)
    _save_matrix(mats["flag_bit_similarity"], labels, OUT_DIR / "tcp_flag_bits_similarity.csv",
                 "TCP flag-bit prevalence similarity (cosine)",
                 FIG_DIR / "tcp_flag_bits_similarity.png", "viridis", vmin=0, vmax=1)

    frames = []
    for field, fm in mats["js_per_field"].items():
        t = _longform(fm, labels, "js_distance")
        t.insert(0, "field", field)
        frames.append(t)
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "jensen_shannon_per_field.csv", index=False)

    # Cosine + Wasserstein support/coverage (consistent-feature policy).
    pd.DataFrame(mats["cosine_feature_support"], index=labels,
                 columns=d["magnitude_features"]).to_csv(OUT_DIR / "cosine_feature_support.csv")
    pd.DataFrame(mats["cosine_pair_coverage"], index=labels, columns=labels).to_csv(
        OUT_DIR / "cosine_pair_coverage.csv")
    with open(OUT_DIR / "cosine_consistent_features.json", "w", encoding="utf-8") as fh:
        json.dump([d["magnitude_features"][p] for p in mats["cosine_consistent_features"]], fh, indent=2)
    frames = []
    for feat, fm in mats["wasserstein_per_feature"].items():
        t = _longform(fm, labels, "wasserstein")
        t.insert(0, "feature", feat)
        frames.append(t)
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "wasserstein_per_feature.csv", index=False)
    pd.DataFrame(mats["wasserstein_feature_support"], index=labels,
                 columns=d.get("dist_features", DIST_FEATURES)).to_csv(OUT_DIR / "wasserstein_feature_support.csv")
    pd.DataFrame(mats["wasserstein_pair_coverage"], index=labels, columns=labels).to_csv(
        OUT_DIR / "wasserstein_pair_coverage.csv")
    frames = []
    for block, bm in mats["relation_block_distance"].items():
        t = _longform(bm, labels, "js_distance")
        t.insert(0, "block", block)
        frames.append(t)
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "relation_block_distance.csv", index=False)
    pd.DataFrame(mats["js_defined_counts"], index=labels, columns=labels).to_csv(OUT_DIR / "js_defined_field_count.csv")
    pd.DataFrame(mats["relation_defined_counts"], index=labels, columns=labels).to_csv(OUT_DIR / "relation_defined_block_count.csv")
    pd.DataFrame(mats["categorical_compression"]).T.to_csv(OUT_DIR / "categorical_compression.csv")
    with open(OUT_DIR / "categorical_universe.json", "w", encoding="utf-8") as fh:
        json.dump({"cat_topk": cfg["cat_topk"],
                   "universe": {f: [str(x) for x in u] for f, u in mats["field_universe"].items()}},
                  fh, indent=2)

    rows = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            rows.append({"class_a": labels[a], "class_b": labels[b],
                         "src_host_jaccard": round(float(mats["host_overlap"][a, b, 0]), 6),
                         "dst_host_jaccard": round(float(mats["host_overlap"][a, b, 1]), 6)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "host_identity_overlap.csv", index=False)

    pd.DataFrame(mats["flag_bit_prevalence"], index=all_classes, columns=FLAG_BIT_NAMES).loc[
        labels].to_csv(OUT_DIR / "tcp_flag_bit_prevalence.csv")
    pd.DataFrame(mats["flag_valid_counts"], index=all_classes, columns=FLAG_MASKS).loc[
        labels].to_csv(OUT_DIR / "tcp_flag_valid_counts.csv")
    pd.DataFrame(mats["flag_support_gaps"], columns=["class", "flag_field"]).to_csv(
        OUT_DIR / "flag_support_gaps.csv", index=False)

    rows = []
    for c, name in enumerate(all_classes):
        for j, feat in enumerate(d["magnitude_features"]):
            if d["c_nan"][c, j] or d["c_inf"][c, j] or d["c_neg"][c, j] or d["c_valid"][c, j]:
                rows.append({"class": name, "feature": feat, "valid": int(d["c_valid"][c, j]),
                             "nan": int(d["c_nan"][c, j]), "inf": int(d["c_inf"][c, j]),
                             "negative": int(d["c_neg"][c, j])})
    pd.DataFrame(rows, columns=["class", "feature", "valid", "nan", "inf", "negative"]).to_csv(
        OUT_DIR / "feature_validity.csv", index=False)
    rows = []
    for c, name in enumerate(all_classes):
        for j, field in enumerate(CATEGORICAL_FIELDS):
            kinds = d["c_cat_kind"][c, j]
            if kinds[1:].sum():
                rows.append({"class": name, "field": field, "nan": int(kinds[1]), "inf": int(kinds[2]),
                             "fractional": int(kinds[3]), "out_of_range": int(kinds[4])})
    pd.DataFrame(rows, columns=["class", "field", "nan", "inf", "fractional", "out_of_range"]).to_csv(
        OUT_DIR / "invalid_categorical_values.csv", index=False)
    rows = []
    for c, name in enumerate(all_classes):
        for k, mask in enumerate(FLAG_MASKS):
            kinds = d["c_flag_kind"][c, k]
            if kinds[1:].sum():
                rows.append({"class": name, "flag_field": mask, "valid": int(kinds[0]),
                             "nan": int(kinds[1]), "inf": int(kinds[2]), "negative": int(kinds[3]),
                             "fractional": int(kinds[4]), "unsupported": int(kinds[5])})
    pd.DataFrame(rows, columns=["class", "flag_field", "valid", "nan", "inf", "negative",
                                "fractional", "unsupported"]).to_csv(
        OUT_DIR / "invalid_flag_values.csv", index=False)

    feas.to_csv(OUT_DIR / "sample_feasibility.csv", index=False)
    sweep.to_csv(OUT_DIR / "window_occupancy.csv", index=False)
    sens.to_csv(OUT_DIR / "margin_gap_sensitivity.csv", index=False)
    window_feas.to_csv(OUT_DIR / "window_feasibility.csv", index=False)

    _rankings(labels, mats["cosine"], "cosine", ascending=False).to_csv(OUT_DIR / "rankings_cosine.csv", index=False)
    _rankings(labels, mats["wasserstein"], "wasserstein", ascending=True).to_csv(OUT_DIR / "rankings_wasserstein.csv", index=False)
    _rankings(labels, mats["jensen_shannon"], "js_distance", ascending=True).to_csv(OUT_DIR / "rankings_js.csv", index=False)
    _rankings(labels, mats["relation_profile"], "relation_similarity", ascending=False).to_csv(OUT_DIR / "rankings_relation.csv", index=False)
    _rankings(labels, mats["flag_bit_similarity"], "flag_bits_similarity", ascending=False).to_csv(OUT_DIR / "rankings_flag_bits.csv", index=False)
    pairs = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            pairs.append({"class_a": labels[a], "class_b": labels[b],
                          "cosine": round(float(mats["cosine"][a, b]), 4),
                          "wasserstein": round(float(mats["wasserstein"][a, b]), 4),
                          "js_distance": round(float(mats["jensen_shannon"][a, b]), 4),
                          "relation_similarity": round(float(mats["relation_profile"][a, b]), 4),
                          "src_host_jaccard": round(float(mats["host_overlap"][a, b, 0]), 4),
                          "dst_host_jaccard": round(float(mats["host_overlap"][a, b, 1]), 4),
                          "flag_bits_similarity": round(float(mats["flag_bit_similarity"][a, b]), 4)})
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(OUT_DIR / "pair_comparison.csv", index=False)

    bundle = {
        "dataset": d.get("dataset_name", ""), "provenance": prov, "config": cfg,
        "n_rows": total_rows, "attack_classes": labels, "split": split_manifest,
        "invalid_value_totals": {
            "nan": int(d["g_nan"].sum()), "inf": int(d["g_inf"].sum()),
            "negative": int(d["g_neg"].sum()),
            "categorical_nan": int(d["g_cat_kind"][:, 1].sum()),
            "categorical_inf": int(d["g_cat_kind"][:, 2].sum()),
            "categorical_fractional": int(d["g_cat_kind"][:, 3].sum()),
            "categorical_out_of_range": int(d["g_cat_kind"][:, 4].sum()),
            "flag_valid": int(d["g_flag_kind"][:, 0].sum()),
            "flag_nan": int(d["g_flag_kind"][:, 1].sum()),
            "flag_inf": int(d["g_flag_kind"][:, 2].sum()),
            "flag_negative": int(d["g_flag_kind"][:, 3].sum()),
            "flag_fractional": int(d["g_flag_kind"][:, 4].sum()),
            "flag_unsupported": int(d["g_flag_kind"][:, 5].sum()),
        },
        "flag_support_gaps": [list(g) for g in mats["flag_support_gaps"]],
        "zero_standardised_profiles": mats["zero_profile_classes"],
        "zero_cosine_pairs": [list(p) for p in mats["zero_cosine_pairs"]],
        "label_counts": {str(k): int(v) for k, v in d["label_counts"].items()},
        "matrices": {
            "standardized_mean_profile_cosine": {"labels": labels, "values": mats["cosine"].round(6).tolist()},
            "wasserstein": {"labels": labels, "values": mats["wasserstein"].round(6).tolist()},
            "jensen_shannon": {"labels": labels, "values": mats["jensen_shannon"].round(6).tolist()},
            "relation_profile": {"labels": labels, "values": mats["relation_profile"].round(6).tolist()},
            "flag_bit_similarity": {"labels": labels, "values": mats["flag_bit_similarity"].round(6).tolist()},
        },
    }
    with open(OUT_DIR / "similarity_results.json", "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2)
    return {"counts_df": counts_df, "pairs_df": pairs_df}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md_table(df, floatfmt="%.3f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = [floatfmt % r[c] if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(d, mats, feas, sweep, sens, window_feas, out, cfg, prov) -> None:
    labels = d["matrix_classes"]
    dataset = d.get("dataset_name", "")
    t70, t85 = d["cuts"]

    def md(mat):
        df = pd.DataFrame(np.round(mat, 3), index=labels, columns=labels)
        df.index.name = "class"
        return df.reset_index()

    pairs = out["pairs_df"]
    most_cos = pairs.sort_values("cosine", ascending=False).head(4)
    most_rel = pairs.sort_values("relation_similarity", ascending=False).head(4)
    most_js = pairs.sort_values("js_distance").head(4)
    cat_invalid = int(d["g_cat_kind"][:, 1:].sum())
    flag_invalid = int(d["g_flag_kind"][:, 1:].sum())
    zero_support = [c for c in labels if int(d["attack_counts"].get(c, 0)) == 0]
    flag_gaps_line = ", ".join(f"{c}/{m}" for c, m in mats["flag_support_gaps"]) or "none"
    zero_profile = mats["zero_profile_classes"]
    label_line = ", ".join(f"{k} -> {v:,}" for k, v in sorted(d["label_counts"].items()))
    train_rows = int(d["ccount_split"][:, 0].sum())

    text = f"""# Step 1 -- Attack-Class Similarity Analysis ({dataset})

*Generated; evidence only. All similarity metrics use the TRAINING partition.*

## 0. Scope, provenance and checks

- **Dataset:** {dataset}, {d['n_rows']:,} flows, sha256 `{prov['data_sha256']}`.
- **Global split (all-flow timestamp quantiles):** train < {t70}, val < {t85}, test >= {t85}.
  Similarity uses **train only** ({train_rows:,} rows); test never influences task-design evidence.
- **Run:** seed {cfg['seed']}, window {cfg['window_ms']} ms, split {cfg['split']},
  floor {cfg['split_floor']}, reservoir {cfg['reservoir_cap']}, cat-topk {cfg['cat_topk']}.
- **Classes compared:** {len(labels)} discovered attack classes.
- **Labels:** {label_line}; Label/Attack mismatches = {d['label_mismatch']} (run stops if > 0).
- **Checks:** zero-support attack classes = {zero_support or 'none'};
  invalid categorical cells = {cat_invalid}; invalid flag cells = {flag_invalid}.
- **Similarity is symmetric by construction; transfer is directional and is NOT tested here.**
- Datasets are analysed separately and same-name labels are not assumed equivalent.

## 1. Attack-class counts and taxonomy

{_md_table(out['counts_df'][out['counts_df']['class'] != BENIGN][['class', 'family', 'flows']])}

## 2. Standardised mean-profile cosine (train-only, magnitude features)

Uses only MAGNITUDE features ({len(d['magnitude_features'])}); identifier codes and
bitmasks are excluded. Zero standardised-deviation profiles: {zero_profile or 'none'}.

{_md_table(md(mats['cosine']))}

Most similar: {', '.join(f"{r.class_a}-{r.class_b} ({r.cosine:.3f})" for r in most_cos.itertuples())}.

## 3. Wasserstein distance (valid samples, train-only)

Mean 1-D Wasserstein over {len(d.get('dist_features', DIST_FEATURES))} continuous features using **valid**
observations only (invalid values are excluded, never zero-filled).

{_md_table(md(mats['wasserstein']))}

## 4. Jensen-Shannon distance (train-only categorical)

A pair averages only over fields valid for both classes; per-field coverage is in
`js_defined_field_count.csv`. Closest: {', '.join(f"{r.class_a}-{r.class_b} ({r.js_distance:.3f})" for r in most_js.itertuples())}.

{_md_table(md(mats['jensen_shannon']))}

## 5. Relation-profile similarity (identity-free, train-only)

{' / '.join(RELATION_BLOCKS)}. These are identity-free structural summaries, **not**
graph topology and **not** learned relation embeddings. Individual block matrices are
the primary evidence (`relation_block_distance.csv`); the average is secondary.

{_md_table(md(mats['relation_profile']))}

## 6. TCP flag bits

Flag values must be finite, integral and 0..255; invalid flags are excluded from bits
and denominators. Missing (class, flag-column) support: {flag_gaps_line}.

## 7. Feasibility (shared global partition)

Support per global split, plus per-window occupancy (`window_feasibility.csv`).

{_md_table(feas[['class', 'flows', 'train_flows', 'val_flows', 'test_flows', 'min_split_flows', 'attack_floor_ok', 'benign_ok_train', 'benign_ok_val', 'benign_ok_test']])}

## 8. Pair comparison

{_md_table(pairs[['class_a', 'class_b', 'cosine', 'wasserstein', 'js_distance', 'relation_similarity', 'src_host_jaccard', 'dst_host_jaccard', 'flag_bits_similarity']], '%.4f')}
"""
    (OUT_DIR / "similarity_report.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    repo = find_repo_root(STEP_DIR)
    parser = argparse.ArgumentParser(description="Step-1 attack-class similarity analysis.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--taxonomy-csv", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=STEP_DIR / "outputs")
    parser.add_argument("--reservoir-cap", type=int, default=20_000)
    parser.add_argument("--window-ms", type=int, default=5_000)
    parser.add_argument("--window-sweep-ms", type=int, nargs="+", default=[1_000, 5_000, 10_000, 30_000, 60_000])
    parser.add_argument("--min-graph-flows", type=int, default=2)
    parser.add_argument("--split", type=float, nargs=2, default=[0.70, 0.85])
    parser.add_argument("--split-floor", type=int, default=1_000)
    parser.add_argument("--min-train-flows", type=int, default=1,
                        help="minimum training flows for a class to enter similarity matrices")
    parser.add_argument("--min-feature-samples", type=int, default=2)
    parser.add_argument("--exclude-features", nargs="*", default=[],
                        help="magnitude feature names to exclude (for ablation; baseline untouched)")
    parser.add_argument("--cat-topk", type=int, default=0, help="0 = full observed support")
    parser.add_argument("--port-topk", type=int, default=None, help="deprecated alias for --cat-topk")
    parser.add_argument("--margin-min", type=float, default=15.0)
    parser.add_argument("--episode-gap-min", type=float, default=15.0)
    parser.add_argument("--sensitivity-gaps-min", type=float, nargs="+", default=[1, 5, 15, 30, 60])
    parser.add_argument("--sensitivity-margins-min", type=float, nargs="+", default=[0, 5, 15, 30])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=250_000)
    parser.add_argument("--max-rows", type=int, default=None, help="smoke-test row cap")
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    if args.data is None:
        rel_path, dataset_name = DATASETS[args.dataset]
        args.data = repo / rel_path
    else:
        dataset_name = Path(args.data).stem
    cat_topk = args.port_topk if args.port_topk is not None else args.cat_topk
    cfg = {
        "reservoir_cap": args.reservoir_cap, "window_ms": args.window_ms,
        "window_sweep_ms": args.window_sweep_ms, "split": tuple(args.split),
        "split_floor": args.split_floor, "min_feature_samples": args.min_feature_samples,
        "min_train_flows": args.min_train_flows, "cat_topk": cat_topk,
        "exclude_features": list(args.exclude_features),
        "margin_min": args.margin_min, "episode_gap_min": args.episode_gap_min,
        "sensitivity_gaps": args.sensitivity_gaps_min,
        "sensitivity_margins": args.sensitivity_margins_min,
        "min_graph_flows": args.min_graph_flows,
        "seed": args.seed, "chunk": args.chunk, "out": str(args.out),
        "dataset": args.dataset, "dataset_name": dataset_name,
        "taxonomy_csv": str(args.taxonomy_csv) if args.taxonomy_csv else None,
        "max_rows": args.max_rows,
    }
    validate_cfg(cfg)
    return args, cfg


def main() -> None:
    global OUT_DIR, FIG_DIR
    args, cfg = parse_args()
    OUT_DIR = args.out
    FIG_DIR = args.out / "figures"
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 120

    if not args.data.exists():
        raise SystemExit(f"dataset not found: {args.data}")
    print(f"[paths] repo root : {find_repo_root(STEP_DIR)}")
    print(f"[paths] dataset   : {cfg['dataset_name']} (key={cfg['dataset']})")
    print(f"[paths] data      : {args.data.resolve()}")
    print(f"[paths] outputs   : {args.out.resolve()}")

    t0 = time.time()
    data_sha256 = "skipped" if args.skip_hash else sha256(args.data)
    prov = {"data_path": str(args.data.resolve()),
            "data_size_mb": round(args.data.stat().st_size / 1e6, 1),
            "data_sha256": data_sha256,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    attack_classes = discover_attack_classes(args.data, cfg)
    print(f"[discover] {len(attack_classes)} attack classes: {attack_classes}")
    cuts = compute_global_split_cuts(args.data, cfg, cfg["window_ms"])
    print(f"[discover] global split cuts (window-aligned): train<{cuts[0]}, val<{cuts[1]}")

    d = extract(args.data, cfg, attack_classes, cuts)
    if d["label_mismatch"] > 0:
        raise SystemExit(f"Label/Attack mismatches: {d['label_mismatch']}. Stopping.")

    # Feasibility first: one unsupported class must not hide the diagnostic.
    supported, scope_df = train_support_scope(d, cfg)
    feas, sweep = build_feasibility(d, cfg)
    if len(supported) < 2:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        feas.to_csv(OUT_DIR / "sample_feasibility.csv", index=False)
        scope_df.to_csv(OUT_DIR / "class_scope.csv", index=False)
        raise SystemExit("fewer than 2 attack classes have training support; "
                         "see sample_feasibility.csv and class_scope.csv")
    d["matrix_classes"] = supported

    flag_gaps = flag_support_gaps(d["c_flag_valid"], d["all_classes"], FLAG_MASKS)
    supported_set = set(supported)
    attack_gaps = [g for g in flag_gaps if g[0] in supported_set]
    if attack_gaps:
        raise SystemExit(f"attack classes with zero valid flag observations: {attack_gaps}. Stopping.")

    mats = build_matrices(d, cfg)
    sens = build_margin_gap_sensitivity(d, cfg)
    window_feas = winmod.window_feasibility(d["ts_parts"], d["all_classes"], d["cuts"],
                                            cfg["window_ms"], cfg["min_graph_flows"])
    out = write_outputs(d, mats, feas, sweep, sens, window_feas, cfg, prov)
    write_report(d, mats, feas, sweep, sens, window_feas, out, cfg, prov)

    print(f"[check] label/attack mismatches : {d['label_mismatch']}")
    print(f"[check] flag support gaps       : {flag_gaps or 'none'}")
    print(f"[done] wrote outputs to {OUT_DIR} in {time.time() - t0:.1f}s")


OUT_DIR = STEP_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"

if __name__ == "__main__":
    main()
