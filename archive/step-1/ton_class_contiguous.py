"""NF-ToN-IoT-v3 under a PER-CLASS contiguous chronological split.

Motivation: a single global chronological split leaves 5 of 9 ToN classes with zero
training flows and none of the supported classes with meaningful test support. Here each
attack class ``c`` gets its own 70/15/15 chronological cut points (60 s aligned), so every
class keeps a temporal train -> val -> test direction while remaining represented.

This is a *different estimand* from a global chronological deployment split: the train
period of one class may follow the test period of another. Do not compare its numbers
directly against the UNSW global-split run.

Safeguards implemented:
  A. 60 s-aligned cuts -> a physical window is never divided between splits for a class.
  B. Benign is matched per window; benign reuse across class tasks and cross-partition
     benign are audited and reported.
  C. Minimum-support / feasibility reporting is kept; unsupported classes are not forced.
  D. Per-class calendar coverage is reported and tasks are ordered by first occurrence.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import attack_class_similarity as acs

ALIGN_MS = 60_000
SPLITS = acs.SPLIT_NAMES


def _snap(x: float) -> int:
    return int(x // ALIGN_MS * ALIGN_MS)


def per_class_cuts(data_path: Path, cfg: dict, classes: list[str]):
    """One pass over FLOW_START+Attack -> per-class 60 s-aligned 70/15/15 cuts."""
    parts = {c: [] for c in classes}
    atk = []
    for chunk in acs._iter_csv(data_path, cfg, usecols=["FLOW_START_MILLISECONDS", "Attack"]):
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        a = chunk["Attack"].to_numpy()
        atk.append(a)
        for c in classes:
            m = a == c
            if m.any():
                parts[c].append(t[m].copy())
    cuts = {}
    win_split = {}
    for c in classes:
        ts = np.sort(np.concatenate(parts[c])) if parts[c] else np.array([], dtype=np.int64)
        if ts.size < 3:
            cuts[c] = None
            continue
        q70, q85 = np.quantile(ts, [0.70, 0.85])
        t70, t85 = _snap(q70), _snap(q85)
        if t70 >= t85:
            t70 = _snap(q70); t85 = min(_snap(q85), int(ts[-1]) // ALIGN_MS * ALIGN_MS + ALIGN_MS)
        cuts[c] = {"t70": int(t70), "t85": int(t85), "n": int(ts.size),
                   "first": int(ts[0]), "last": int(ts[-1])}
        wu, inv = np.unique((ts // ALIGN_MS).astype(np.int64), return_inverse=True)
        wts = (wu * ALIGN_MS)
        win_split[c] = {int(w): (0 if t < t70 else (1 if t < t85 else 2))
                        for w, t in zip(wu.tolist(), wts.tolist())}
    return cuts, win_split


def _merged_window_split(win_split):
    """window -> split array (classes are temporally disjoint, so windows do not conflict)."""
    allkeys = [w for d in win_split.values() if d for w in d]
    if not allkeys:
        return None
    lo, hi = min(allkeys), max(allkeys)
    arr = np.full(hi - lo + 1, -1, dtype=np.int8)
    for d in win_split.values():
        for w, s in d.items():
            arr[w - lo] = s
    return lo, arr


def build(data_path: Path, cfg: dict, classes: list[str], cuts: dict, win_split=None) -> dict:
    names = [acs.BENIGN] + list(classes)
    wsplit = _merged_window_split(win_split or {})
    wsplit_lo, wsplit_arr = wsplit if wsplit else (0, None)
    code_of = {n: i for i, n in enumerate(names)}
    K = len(names)
    features = acs.MAGNITUDE_FEATURES
    F = len(features)
    dist_features = [f for f in acs.DIST_FEATURES if f in features]
    n_cat, n_flag, n_bits = len(acs.CATEGORICAL_FIELDS), len(acs.FLAG_MASKS), len(acs.FLAG_BIT_NAMES)
    cap = cfg["reservoir_cap"]
    rng = np.random.default_rng(cfg["seed"])

    g_valid = np.zeros(F, dtype=np.int64); g_sum = np.zeros(F); g_sumsq = np.zeros(F)
    c_valid = np.zeros((K, F), dtype=np.int64); c_sum = np.zeros((K, F)); c_sumsq = np.zeros((K, F))
    res = np.full((K, cap, F), np.nan, dtype=np.float32); seen = np.zeros(K, dtype=np.int64)
    cat_ct = [[Counter() for _ in acs.CATEGORICAL_FIELDS] for _ in range(K)]
    src_ct = [Counter() for _ in range(K)]; dst_ct = [Counter() for _ in range(K)]
    pair_ct = [Counter() for _ in range(K)]
    flagbit = np.zeros((K, n_bits), dtype=np.int64); c_flag_valid = np.zeros((K, n_flag), dtype=np.int64)
    ccount = np.zeros((K, 3), dtype=np.int64)
    # benign audit
    benign_split = np.zeros(3, dtype=np.int64)
    benign_win_train: dict[int, set] = defaultdict(set)
    benign_flow_win: dict[int, int] = defaultdict(int)  # window -> benign flows

    t70s = np.array([cuts[c]["t70"] for c in classes if cuts[c]], dtype=np.int64)

    n_rows = 0
    for chunk in acs._iter_csv(data_path, cfg):
        n_rows += len(chunk)
        a = chunk["Attack"].to_numpy()
        codes = chunk["Attack"].map(code_of).to_numpy(dtype=np.int16)
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        X = chunk[features].to_numpy(dtype=np.float64)
        valid = np.isfinite(X) & (X >= 0.0)
        L = np.log1p(np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None))
        L[~valid] = np.nan
        # per-class split assignment for attack rows
        sp_att = np.full(len(chunk), -1, dtype=np.int8)
        for ci, c in enumerate(classes, start=1):
            cut = cuts.get(c)
            if not cut:
                continue
            m = codes == ci
            if m.any():
                sp_att[m] = np.where(t[m] < cut["t70"], 0, np.where(t[m] < cut["t85"], 1, 2))
        is_benign = codes == 0
        # benign matched by WINDOW MEMBERSHIP: a benign window takes the split of the class it
        # contains; windows with no class are unused. This keeps val/test benign out of g_*.
        bsplit = np.full(len(chunk), -1, dtype=np.int8)
        if wsplit_arr is not None:
            widx = (t // ALIGN_MS).astype(np.int64) - wsplit_lo
            ok = (widx >= 0) & (widx < wsplit_arr.size)
            bsplit[ok] = wsplit_arr[widx[ok]]
        # global standardisation reference = any train flow (attack-train or benign-train)
        gmask = ((sp_att == 0) | (is_benign & (bsplit == 0)))
        if gmask.any():
            Lt = L[gmask]
            g_valid += np.isfinite(Lt).sum(axis=0)
            g_sum += np.nansum(Lt, axis=0)
            g_sumsq += np.nansum(Lt * Lt, axis=0)
        if is_benign.any():
            bm = is_benign & (bsplit >= 0)
            if bm.any():
                benign_split += np.bincount(bsplit[bm], minlength=3)
                bidx = np.flatnonzero(bm)
                bw = (t[bidx] // ALIGN_MS).astype(np.int64)
                u, inv = np.unique(bw, return_inverse=True)
                cnts = np.bincount(inv, minlength=u.size)
                for b, cn in zip(u.tolist(), cnts.tolist()):
                    benign_flow_win[int(b)] += int(cn)

        cat_codes, cat_ok = {}, {}
        for j, field in enumerate(acs.CATEGORICAL_FIELDS):
            v, cc, _ = acs.categorical_field_codes(field, chunk[field].to_numpy(dtype=np.float64))
            cat_codes[field] = cc; cat_ok[field] = v
        bitmat = np.zeros((len(chunk), n_bits), dtype=bool)
        fval = np.zeros((len(chunk), n_flag), dtype=bool)
        col = 0
        for k, mask in enumerate(acs.FLAG_MASKS):
            v, cc, _ = acs.flag_value_report(chunk[mask].to_numpy(dtype=np.float64))
            fval[:, k] = v
            for _, bit in acs.FLAG_BITS:
                bitmat[:, col] = v & ((cc & bit) != 0); col += 1
        src = chunk[acs.HOST_SRC].to_numpy(); dst = chunk[acs.HOST_DST].to_numpy()

        for ci in range(1, K):
            m = codes == ci
            if not m.any():
                continue
            sp = sp_att[m]
            ccount[ci] += np.bincount(sp, minlength=3)[:3] if sp.min() >= 0 else np.zeros(3, dtype=np.int64)
            tr = np.flatnonzero(m)[sp == 0]
            if tr.size == 0:
                continue
            c_valid[ci] += np.isfinite(L[tr]).sum(axis=0)
            c_sum[ci] += np.nansum(L[tr], axis=0)
            c_sumsq[ci] += np.nansum(L[tr] * L[tr], axis=0)
            src_ct[ci].update(src[tr].tolist()); dst_ct[ci].update(dst[tr].tolist())
            pair_ct[ci].update(zip(src[tr].tolist(), dst[tr].tolist()))
            for j, field in enumerate(acs.CATEGORICAL_FIELDS):
                vmask = cat_ok[field][tr]
                cat_ct[ci][j].update(cat_codes[field][tr][vmask].tolist())
            flagbit[ci] += bitmat[tr].sum(axis=0)
            c_flag_valid[ci] += fval[tr].sum(axis=0)
            seen[ci] = acs._reservoir_update(res[ci], int(seen[ci]),
                                             np.asarray(L[tr], dtype=np.float32), rng, cap)
            for b in np.unique((t[tr] // ALIGN_MS).astype(np.int64)):
                benign_win_train[int(b)].add(ci)
        if n_rows % 2_000_000 < cfg["chunk"]:
            print(f"  scanned {n_rows:,} rows ...")
    print(f"[contig] done: {n_rows:,} rows")
    d = {"n_rows": n_rows, "dataset_name": cfg["dataset_name"], "all_classes": names,
         "magnitude_features": features, "dist_features": dist_features, "F": F, "K": K,
         "g_valid": g_valid, "g_sum": g_sum, "g_sumsq": g_sumsq,
         "c_valid": c_valid, "c_sum": c_sum, "c_sumsq": c_sumsq,
         "res": res, "seen": seen, "cat_ct": cat_ct, "src_ct": src_ct, "dst_ct": dst_ct,
         "pair_ct": pair_ct, "flagbit": flagbit, "c_flag_valid": c_flag_valid,
         "ccount_split": ccount, "benign_split": benign_split}
    extra = {"benign_win_train": benign_win_train, "benign_flow_win": benign_flow_win}
    return d, extra


def write_outputs(d, extra, cuts, cfg, args, prov):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    names = d["all_classes"]; classes = names[1:]
    M = len(classes)
    cc = d["ccount_split"]
    floor = cfg["split_floor"]

    # attack_class_counts
    rows = []
    for i, c in enumerate(names):
        flows = int(d["benign_split"].sum()) if c == acs.BENIGN else int(cc[i].sum())
        rows.append({"class": c, "is_attack": c != acs.BENIGN, "flows": flows})
    pd.DataFrame(rows).to_csv(out / "attack_class_counts.csv", index=False)

    # class_temporal_coverage
    cov = []
    for c in classes:
        cut = cuts[c]
        cov.append({"class": c, "first_occurrence": pd.to_datetime(cut["first"], unit="ms", utc=True),
                    "last_occurrence": pd.to_datetime(cut["last"], unit="ms", utc=True),
                    "train_end": pd.to_datetime(cut["t70"], unit="ms", utc=True),
                    "val_end": pd.to_datetime(cut["t85"], unit="ms", utc=True),
                    "span_hours": round((cut["last"] - cut["first"]) / 3.6e6, 2)})
    cov = pd.DataFrame(cov).sort_values("first_occurrence")
    cov.to_csv(out / "class_temporal_coverage.csv", index=False)

    # class_scope + sample_feasibility
    scope, feas = [], []
    for ci, c in enumerate(classes, start=1):
        tr = int(cc[ci, 0]); va = int(cc[ci, 1]); te = int(cc[ci, 2])
        status = ("train_supported" if tr >= floor else
                  "low_train_support" if tr > 0 else "zero_train_support")
        scope.append({"class": c, "train_flows": tr, "status": status})
        feas.append({"class": c, "flows": tr + va + te, "train_flows": tr, "val_flows": va,
                     "test_flows": te, "min_split_flows": min(tr, va, te),
                     "attack_floor_ok": min(tr, va, te) >= floor, "train_supported": tr >= floor})
    pd.DataFrame(scope).to_csv(out / "class_scope.csv", index=False)
    pd.DataFrame(feas).to_csv(out / "sample_feasibility.csv", index=False)

    # split_manifest (per-class cuts) + benign audit
    t70s = [cuts[c]["t70"] for c in classes]; t85s = [cuts[c]["t85"] for c in classes]
    win_shared = {b: sorted(v) for b, v in extra["benign_win_train"].items() if len(v) > 1}
    audit = {
        "benign_flows_by_split": {SPLITS[i]: int(d["benign_split"][i]) for i in range(3)},
        "windows_train_for_multiple_classes": len(win_shared),
        "benign_flows_in_multi_class_train_windows": int(sum(
            extra["benign_flow_win"].get(b, 0) for b in win_shared)),
        "note": "benign is matched by window membership: a benign window takes the split of the "
                "class it contains; windows with no class are unused, so val/test benign never "
                "enters training statistics.",
    }
    split_manifest = {
        "mode": "per_class_contiguous_chronological",
        "align_ms": ALIGN_MS, "split_quantiles": list(cfg["split"]),
        "estimand": "per-class chronological; NOT a globally forward-in-time split",
        "per_class_cuts": {c: {"t70": cuts[c]["t70"], "t85": cuts[c]["t85"]} for c in classes},
        "class_order_by_first_occurrence": cov["class"].tolist(),
        "benign_audit": audit,
        "dataset": d["dataset_name"], "sha256": prov["data_sha256"], "rows": d["n_rows"],
    }
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, default=str),
                                             encoding="utf-8")

    # similarity matrices via the shared builder
    supported = [c for c in classes if cc[names.index(c), 0] >= floor]
    if len(supported) < 2:
        print(f"[contig] only {len(supported)} train-supported classes; writing split-only outputs")
        (out / "similarity_results.json").write_text(json.dumps({"classes": supported}), encoding="utf-8")
        return
    d["matrix_classes"] = supported
    mats = acs.build_matrices(d, cfg)
    labels = supported
    def mat_to_dict(m):
        return {labels[i]: {labels[j]: round(float(m[i, j]), 6) for j in range(len(labels))}
                for i in range(len(labels))}
    results = {"classes": labels, "per_class_split_flows": feas,
               "cosine": mat_to_dict(mats["cosine"]),
               "wasserstein": mat_to_dict(mats["wasserstein"]),
               "jensen_shannon": mat_to_dict(mats["jensen_shannon"]),
               "relation_profile": mat_to_dict(mats["relation_profile"]),
               "flag_bit_similarity": mat_to_dict(mats["flag_bit_similarity"]),
               "host_overlap_note": "identity overlap is supplementary; relation profiles are identity-free"}
    (out / "similarity_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _report(out / "similarity_report.md", d, mats, labels, cov, feas, scope, audit, prov, cfg)
    print(f"[contig] wrote {out}")


def _report(path, d, mats, labels, cov, feas, scope, audit, prov, cfg):
    def tbl(m):
        lines = ["| class | " + " | ".join(labels) + " |",
                 "| --- | " + " | ".join("---" for _ in labels) + " |"]
        for i, a in enumerate(labels):
            lines.append(f"| {a} | " + " | ".join(f"{m[i, j]:.3f}" for j in range(len(labels))) + " |")
        return "\n".join(lines)
    L = [f"# Step 1.4 -- NF-ToN-IoT-v3, per-class contiguous chronological split", "",
         "*Different estimand from a global chronological split - see split_manifest.json.*", "",
         f"- Dataset sha256 `{prov['data_sha256']}`; rows {d['n_rows']:,}; alignment {ALIGN_MS//1000}s.",
         f"- Classes: {', '.join(labels)} (train-supported, >= {cfg['split_floor']}).",
         f"- Split-only classes: {', '.join(s['class'] for s in scope if s['status'] != 'train_supported')}.",
         f"- Benign by split: {audit['benign_flows_by_split']}.",
         f"- Windows in >1 class train set: {audit['windows_train_for_multiple_classes']} "
         f"({audit['benign_flows_in_multi_class_train_windows']:,} benign flows, shared-normal exposure).",
         "", "## Per-class calendar coverage", "",
         "| class | first | last | train_end | val_end | span_h |",
         "| --- | --- | --- | --- | --- | --- |"]
    for r in cov.to_dict("records"):
        L.append(f"| {r['class']} | {r['first_occurrence']} | {r['last_occurrence']} | "
                 f"{r['train_end']} | {r['val_end']} | {r['span_hours']} |")
    L += ["", "## Feasibility (per-class split)", "",
          "| class | flows | train | val | test | floor_ok |", "| --- | --- | --- | --- | --- | --- |"]
    for r in feas:
        L.append(f"| {r['class']} | {r['flows']:,} | {r['train_flows']:,} | {r['val_flows']:,} | "
                 f"{r['test_flows']:,} | {r['attack_floor_ok']} |")
    for name, key, note in [("Standardised mean-profile cosine (train-only)", "cosine", ""),
                            ("Wasserstein distance (train-only)", "wasserstein", ""),
                            ("Jensen-Shannon distance (train-only)", "jensen_shannon", ""),
                            ("Relation-profile similarity (identity-free)", "relation_profile", ""),
                            ("TCP flag-bit similarity", "flag_bit_similarity", "")]:
        L += ["", f"## {name}", "", tbl(mats[key])]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def parse_args():
    repo = acs.find_repo_root(acs.STEP_DIR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(acs.DATASETS), default="ton")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "ton_class_contiguous")
    ap.add_argument("--split", type=float, nargs=2, default=[0.70, 0.85])
    ap.add_argument("--split-floor", type=int, default=1000)
    ap.add_argument("--reservoir-cap", type=int, default=20_000)
    ap.add_argument("--cat-topk", type=int, default=0)
    ap.add_argument("--min-feature-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--skip-hash", action="store_true")
    a = ap.parse_args()
    if a.data is None:
        a.data = repo / acs.DATASETS[a.dataset][0]
    return a


def main():
    args = parse_args()
    cfg = {"split": tuple(args.split), "split_floor": args.split_floor,
           "reservoir_cap": args.reservoir_cap, "cat_topk": args.cat_topk,
           "min_feature_samples": args.min_feature_samples, "seed": args.seed,
           "chunk": args.chunk, "max_rows": args.max_rows,
           "dataset_name": acs.DATASETS.get(args.dataset, (None, "dataset"))[1]}
    print(f"[contig] data {args.data.resolve()}")
    sha = "skipped" if args.skip_hash else acs.sha256(args.data)
    classes = acs.discover_attack_classes(args.data, cfg)
    print(f"[contig] {len(classes)} attack classes: {classes}")
    cuts, win_split = per_class_cuts(args.data, cfg, classes)
    print("[contig] per-class cuts ready")
    d, extra = build(args.data, cfg, classes, cuts, win_split)
    prov = {"data_path": str(args.data.resolve()), "data_sha256": sha,
            "data_size_mb": round(args.data.stat().st_size / 1e6, 1),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_outputs(d, extra, cuts, cfg, args, prov)


if __name__ == "__main__":
    main()
