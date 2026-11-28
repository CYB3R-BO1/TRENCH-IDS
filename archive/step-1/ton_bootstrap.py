"""Step 1.4 - ToN-IoT temporal block bootstrap (within-class blocks).

Classes occupy disjoint periods, so resampling is done WITHIN each class's own contiguous
training sequence - classes are never concatenated and never share blocks. Primary block
length 30 min, sensitivity 60 min. Training-only normalisation is recomputed per replicate.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

import attack_class_similarity as acs

FINE_MS = 60_000
METRICS = ["cosine", "wasserstein", "jensen_shannon", "relation_profile", "flag_bit_similarity"]
SIM = {"cosine": True, "wasserstein": False, "jensen_shannon": False,
       "relation_profile": True, "flag_bit_similarity": True}
WASS_CAP = 2000
BLOCK_MIN = 30  # cache granularity (blocks of 30 min)


# --------------------------------------------------------------------------- #
def load_cuts(manifest: Path):
    m = json.loads(manifest.read_text(encoding="utf-8"))
    return {c: (v["t70"], v["t85"]) for c, v in m["per_class_cuts"].items()}


def window_map(data_path, cfg, classes, cuts):
    """window -> (class idx, train?) for every window containing a class."""
    win_cls = {}
    first = {}
    for chunk in acs._iter_csv(data_path, cfg, usecols=["FLOW_START_MILLISECONDS", "Attack"]):
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        a = chunk["Attack"].to_numpy()
        for ci, c in enumerate(classes, start=1):
            m = a == c
            if m.any():
                u = np.unique((t[m] // FINE_MS).astype(np.int64))
                for w in u.tolist():
                    win_cls[int(w)] = ci
                first[c] = min(first.get(c, 10 ** 18), int(u[0]))
    return win_cls, first


def build_cache(data_path, cfg, classes, cuts, win_cls, first):
    F = len(acs.MAGNITUDE_FEATURES)
    dist_pos = [acs.MAGNITUDE_FEATURES.index(f) for f in acs.DIST_FEATURES]
    n_cat, n_flag, n_bits = len(acs.CATEGORICAL_FIELDS), len(acs.FLAG_MASKS), len(acs.FLAG_BIT_NAMES)
    cap = cfg["reservoir_cap"]
    rng = np.random.default_rng(cfg["seed"])
    KF = BLOCK_MIN  # 30 fine windows per 30-min block
    cache: dict[tuple, dict] = {}
    first_w = {c: first.get(c, cuts[c][0]) // FINE_MS for c in classes}
    win_lo = min(win_cls) if win_cls else 0
    win_arr = np.zeros((max(win_cls) - win_lo + 1) if win_cls else 1, dtype=np.int16)
    for w, ci in win_cls.items():
        win_arr[w - win_lo] = ci

    def get(ci, b):
        k = (ci, b)
        if k not in cache:
            cache[k] = {"cv": np.zeros(F, dtype=np.int64), "cs": np.zeros(F), "cq": np.zeros(F),
                        "res": np.full((cap, F), np.nan, dtype=np.float32), "seen": 0,
                        "cat": [Counter() for _ in range(n_cat)],
                        "src": Counter(), "dst": Counter(), "pair": Counter(),
                        "flagbit": np.zeros(n_bits, dtype=np.int64),
                        "flag_valid": np.zeros(n_flag, dtype=np.int64),
                        "bgv": np.zeros(F, dtype=np.int64), "bgs": np.zeros(F), "bgq": np.zeros(F)}
        return cache[k]

    n_rows = 0
    for chunk in acs._iter_csv(data_path, cfg):
        n_rows += len(chunk)
        a = chunk["Attack"].to_numpy()
        codes = chunk["Attack"].map({c: i for i, c in enumerate([acs.BENIGN] + list(classes))}).to_numpy(dtype=np.int16)
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        X = chunk[acs.MAGNITUDE_FEATURES].to_numpy(dtype=np.float64)
        valid = np.isfinite(X) & (X >= 0.0)
        L = np.log1p(np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None))
        L[~valid] = np.nan
        wid = (t // FINE_MS).astype(np.int64)
        # attack-train rows only (encodings are needed only for these)
        atk = np.zeros(len(chunk), dtype=bool)
        for ci, c in enumerate(classes, start=1):
            atk |= (codes == ci) & (t < cuts[c][0])
        # benign magnitude, grouped by the class whose train window it belongs to (vectorised)
        if (codes == 0).any():
            br = np.flatnonzero(codes == 0)
            w = wid[br] - win_lo
            okb = (w >= 0) & (w < win_arr.size)
            cls_arr = np.zeros(br.size, dtype=np.int16)
            cls_arr[okb] = win_arr[w[okb]]
            for ci in np.unique(cls_arr):
                if ci == 0:
                    continue
                c = classes[int(ci) - 1]
                sel = br[cls_arr == ci]
                sel = sel[t[sel] < cuts[c][0]]
                if sel.size == 0:
                    continue
                b = ((wid[sel] - first_w[c]) // KF).astype(np.int64)
                order = np.argsort(b, kind="stable"); bs = b[order]
                uq, st = np.unique(bs, return_index=True); en = np.append(st[1:], bs.size)
                for i, bb in enumerate(uq):
                    rr = sel[order[st[i]:en[i]]]
                    Lr = L[rr]; rec = get(int(ci), int(bb))
                    rec["bgv"] += np.isfinite(Lr).sum(axis=0)
                    rec["bgs"] += np.nansum(Lr, axis=0)
                    rec["bgq"] += np.nansum(Lr * Lr, axis=0)
        if atk.any():
            sub = np.flatnonzero(atk)
            cat_codes, cat_ok = {}, {}
            for j, field in enumerate(acs.CATEGORICAL_FIELDS):
                v, cc, _ = acs.categorical_field_codes(field, chunk[field].to_numpy(dtype=np.float64)[sub])
                cat_codes[field] = cc; cat_ok[field] = v
            bitmat = np.zeros((sub.size, n_bits), dtype=bool); fval = np.zeros((sub.size, n_flag), dtype=bool)
            col = 0
            for k, mask in enumerate(acs.FLAG_MASKS):
                v, cc, _ = acs.flag_value_report(chunk[mask].to_numpy(dtype=np.float64)[sub])
                fval[:, k] = v
                for _, bit in acs.FLAG_BITS:
                    bitmat[:, col] = v & ((cc & bit) != 0); col += 1
            src = chunk[acs.HOST_SRC].to_numpy()[sub]; dst = chunk[acs.HOST_DST].to_numpy()[sub]
            subcode = codes[sub]; subwid = wid[sub]
            for ci, c in enumerate(classes, start=1):
                m = subcode == ci
                if not m.any():
                    continue
                pos = np.flatnonzero(m)
                b = ((subwid[pos] - first_w[c]) // KF).astype(np.int64)
                order = np.argsort(b, kind="stable"); bs = b[order]
                uq, st = np.unique(bs, return_index=True); en = np.append(st[1:], bs.size)
                for i, bb in enumerate(uq):
                    rr = pos[order[st[i]:en[i]]]
                    rec = get(ci, int(bb))
                    Lr = L[sub[rr]]
                    rec["cv"] += np.isfinite(Lr).sum(axis=0)
                    rec["cs"] += np.nansum(Lr, axis=0)
                    rec["cq"] += np.nansum(Lr * Lr, axis=0)
                    for j, field in enumerate(acs.CATEGORICAL_FIELDS):
                        ok = cat_ok[field][rr]
                        if ok.any():
                            rec["cat"][j].update(cat_codes[field][rr][ok].tolist())
                    rec["src"].update(src[rr].tolist()); rec["dst"].update(dst[rr].tolist())
                    rec["pair"].update(zip(src[rr].tolist(), dst[rr].tolist()))
                    rec["flagbit"] += bitmat[rr].sum(axis=0)
                    rec["flag_valid"] += fval[rr].sum(axis=0)
                    rec["seen"] = acs._reservoir_update(rec["res"], int(rec["seen"]),
                                                        np.asarray(Lr, dtype=np.float32), rng, cap)
        if n_rows % 4_000_000 < cfg["chunk"]:
            print(f"  scanned {n_rows:,} rows ...", flush=True)
    print(f"[boot] cache done: {n_rows:,} rows, {len(cache)} class-blocks")
    return cache, dist_pos


def reference(cache, classes):
    """Full-train reference structures: universe, host padding, consistent features, class c_valid."""
    full_cat = {}
    full_src = Counter(); full_dst = Counter(); full_pair = Counter()
    c_valid = {}
    Lhost = Lpair = 1
    for ci, c in enumerate(classes, start=1):
        cv = np.zeros(len(acs.MAGNITUDE_FEATURES), dtype=np.int64)
        cat = [Counter() for _ in acs.CATEGORICAL_FIELDS]
        src, dst, pair = Counter(), Counter(), Counter()
        for (cj, b), rec in cache.items():
            if cj != ci:
                continue
            cv += rec["cv"]
            for j in range(len(acs.CATEGORICAL_FIELDS)):
                cat[j].update(rec["cat"][j])
            src.update(rec["src"]); dst.update(rec["dst"]); pair.update(rec["pair"])
        c_valid[c] = cv
        full_cat[c] = cat
        Lhost = max(Lhost, len(src), len(dst)); Lpair = max(Lpair, len(pair))
    universe = {}
    for j, field in enumerate(acs.CATEGORICAL_FIELDS):
        glob = Counter()
        for c in classes:
            glob.update(full_cat[c][j])
        universe[field] = list(glob.keys())
    cv_all = np.vstack([c_valid[c] for c in classes])
    consistent = [p for p in range(len(acs.MAGNITUDE_FEATURES)) if int(cv_all[:, p].min()) >= 2]
    return {"universe": universe, "L_host": Lhost, "L_pair": Lpair,
            "consistent": consistent, "c_valid": c_valid, "cat_idx": {f: i for i, f in enumerate(acs.CATEGORICAL_FIELDS)}}


def densify(cache, K=256):
    """Replace per-block categorical Counters with dense vectors over a top-K universe (+other).

    Tractability approximation for the bootstrap only: categories outside the top-K are pooled
    into the trailing 'other' slot. Keeps per-replicate pooling as numpy sums.
    """
    universes = {}
    for j, field in enumerate(acs.CATEGORICAL_FIELDS):
        glob = Counter()
        for rec in cache.values():
            glob.update(rec["cat"][j])
        uni = [c for c, _ in glob.most_common(K)]
        idx = {c: i for i, c in enumerate(uni)}
        for rec in cache.values():
            cnt = rec["cat"][j]
            arr = np.zeros(len(uni) + 1, dtype=np.int64)
            tot = 0
            for k, v in cnt.items():
                tot += v
                i = idx.get(k)
                if i is not None:
                    arr[i] += v
            arr[-1] = tot - int(arr[:-1].sum())
            rec.setdefault("vec", {})[field] = arr
        universes[field] = uni
    for rec in cache.values():
        rec["cat"] = None
    return universes


def groups_for(cache, ci, block_min):
    """group index -> list of 30-min block keys for a class at a given block length."""
    ratio = block_min // BLOCK_MIN
    g = defaultdict(list)
    for (cj, b) in cache:
        if cj == ci:
            g[b // ratio].append(b)
    return [v for _, v in sorted(g.items())]


def merge(recs, F, n_cat, n_flag, n_bits):
    out = {"cv": np.zeros(F, dtype=np.int64), "cs": np.zeros(F), "cq": np.zeros(F),
           "src": Counter(), "dst": Counter(), "pair": Counter(),
           "flagbit": np.zeros(n_bits, dtype=np.int64), "flag_valid": np.zeros(n_flag, dtype=np.int64),
           "bgv": np.zeros(F, dtype=np.int64), "bgs": np.zeros(F), "bgq": np.zeros(F),
           "res": [], "vec": None}
    for r in recs:
        out["cv"] += r["cv"]; out["cs"] += r["cs"]; out["cq"] += r["cq"]
        out["bgv"] += r["bgv"]; out["bgs"] += r["bgs"]; out["bgq"] += r["bgq"]
        out["src"].update(r["src"]); out["dst"].update(r["dst"]); out["pair"].update(r["pair"])
        out["flagbit"] += r["flagbit"]; out["flag_valid"] += r["flag_valid"]
        n = int(r["seen"]); out["res"].append(r["res"][:n])
        if out["vec"] is None:
            out["vec"] = {f: v.copy() for f, v in r["vec"].items()}
        else:
            for f, v in r["vec"].items():
                out["vec"][f] += v
    if out["vec"] is None:
        out["vec"] = {}
    return out


def _pairwise_js(items, M):
    """Mean pairwise JS over items where both classes are defined; NaN where none are."""
    num = np.zeros((M, M)); den = np.zeros((M, M), dtype=np.int64)
    for _name, vecs, defined in items:
        for i in range(M):
            if not defined[i]:
                continue
            for j in range(i + 1, M):
                if not defined[j]:
                    continue
                d = float(jensenshannon(vecs[i], vecs[j], base=2))
                num[i, j] += d; num[j, i] += d; den[i, j] += 1; den[j, i] += 1
    out = np.full((M, M), np.nan)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def replicate(cache, classes, ref, gs, sel, rng, dist_pos, F, n_cat, n_flag, n_bits):
    recs = {c: merge([cache[(ci, b)] for i in sel[c] for b in gs[c][i]], F, n_cat, n_flag, n_bits)
            for ci, c in enumerate(classes, start=1)}
    # global normalisation from all resampled attack-train + benign-train flows
    gv = np.zeros(F, dtype=np.int64); gs_ = np.zeros(F); gq = np.zeros(F)
    for c in classes:
        gv += recs[c]["cv"] + recs[c]["bgv"]; gs_ += recs[c]["cs"] + recs[c]["bgs"]; gq += recs[c]["cq"] + recs[c]["bgq"]
    gmean = np.divide(gs_, gv, out=np.zeros(F), where=gv > 0)
    gvar = np.divide(gq, gv, out=np.zeros(F), where=gv > 0) - gmean ** 2
    gstd = np.sqrt(np.clip(gvar, 0.0, None)); gstd[gstd == 0] = 1.0
    M = len(classes)
    cmean = np.vstack([np.divide(recs[c]["cs"], recs[c]["cv"], out=np.full(F, np.nan),
                                 where=recs[c]["cv"] > 0) for c in classes])
    z = (cmean - gmean) / gstd
    cons = [p for p in ref["consistent"] if np.all(np.isfinite(z[:, p]))]
    cos = acs.cosine_matrix(z[:, cons]) if len(cons) >= 2 else np.zeros((M, M))

    wnum = np.zeros((M, M)); wden = np.zeros((M, M), dtype=np.int64)
    for fpos in dist_pos:
        smp = []
        for c in classes:
            arr = recs[c]["res"]
            a = np.concatenate([x[:, fpos] for x in arr]) if arr else np.zeros(0)
            a = a[np.isfinite(a)]
            smp.append((a - gmean[fpos]) / gstd[fpos])
        for i in range(M):
            for j in range(i + 1, M):
                sa, sb = smp[i], smp[j]
                if sa.size >= 2 and sb.size >= 2:
                    if sa.size > WASS_CAP:
                        sa = rng.choice(sa, WASS_CAP, replace=False)
                    if sb.size > WASS_CAP:
                        sb = rng.choice(sb, WASS_CAP, replace=False)
                    w = float(wasserstein_distance(sa, sb))
                    wnum[i, j] += w; wnum[j, i] += w; wden[i, j] += 1; wden[j, i] += 1
    wass = np.divide(wnum, wden, out=np.zeros((M, M)), where=wden > 0)

    cat_items = []
    for field in acs.CATEGORICAL_FIELDS:
        vecs = []
        for c in classes:
            v = recs[c]["vec"].get(field)
            vecs.append((v / v.sum()) if (v is not None and v.sum() > 0) else np.zeros(1))
        cat_items.append((field, vecs, [v.sum() > 0 for v in vecs]))
    js = _pairwise_js(cat_items, M)

    def blk(rec, block):
        if block == "source_host_degree":
            return acs.sorted_share_distribution(rec["src"], ref["L_host"])
        if block == "dest_host_degree":
            return acs.sorted_share_distribution(rec["dst"], ref["L_host"])
        if block == "host_pair_degree":
            return acs.sorted_share_distribution(rec["pair"], ref["L_pair"])
        f = block.split("flow_to_")[1]
        field = {"src_port": "L4_SRC_PORT", "dst_port": "L4_DST_PORT",
                 "protocol": "PROTOCOL", "service": "L7_PROTO"}[f]
        v = rec["vec"].get(field)
        return (v / v.sum()) if (v is not None and v.sum() > 0) else np.zeros(1)
    rel_items = []
    for b in acs.RELATION_BLOCKS:
        vecs = [blk(recs[c], b) for c in classes]
        rel_items.append((b, vecs, [v.sum() > 0 for v in vecs]))
    struct = _pairwise_js(rel_items, M)
    rel = np.clip(1.0 - struct, 0.0, 1.0)
    rel = np.where(np.isfinite(struct), rel, np.nan)

    prev = np.vstack([acs.flag_bit_prevalence(recs[c]["flagbit"][None, :],
                                              recs[c]["flag_valid"][None, :])[0][0] for c in classes])
    flags = acs.cosine_matrix(prev)
    return {"cosine": cos, "wasserstein": wass, "jensen_shannon": js,
            "relation_profile": rel, "flag_bit_similarity": flags}


def run_length(cache, classes, ref, dist_pos, F, n_cat, n_flag, n_bits, block_min, B, seed):
    gs = {c: groups_for(cache, ci, block_min) for ci, c in enumerate(classes, start=1)}
    rng = np.random.default_rng(seed)
    store = {m: [] for m in METRICS}
    for _ in range(B):
        sel = {c: (list(rng.integers(0, len(gs[c]), size=len(gs[c]))) if len(gs[c]) > 0 else [])
               for c in classes}
        mats = replicate(cache, classes, ref, gs, sel, rng, dist_pos, F, n_cat, n_flag, n_bits)
        for m in METRICS:
            store[m].append(mats[m])
    return np.array([store[m] for m in METRICS]), {c: len(gs[c]) for c in classes}


def main():
    repo = acs.find_repo_root(acs.STEP_DIR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(acs.DATASETS), default="ton")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--manifest", type=Path,
                    default=acs.STEP_DIR / "outputs" / "ton_class_contiguous_benignmatched" / "split_manifest.json")
    ap.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "ton_uncertainty" / "bootstrap")
    ap.add_argument("--block-minutes", type=int, nargs="+", default=[30, 60])
    ap.add_argument("--primary-minutes", type=int, default=30)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--reservoir-cap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--skip-hash", action="store_true")
    a = ap.parse_args()
    if a.data is None:
        a.data = repo / acs.DATASETS[a.dataset][0]
    cfg = {"chunk": a.chunk, "max_rows": a.max_rows, "reservoir_cap": a.reservoir_cap, "seed": a.seed}
    cuts = load_cuts(a.manifest)
    classes = list(cuts)
    sha = "skipped" if a.skip_hash else acs.sha256(a.data)
    win_cls, first = window_map(a.data, cfg, classes, cuts)
    cache, dist_pos = build_cache(a.data, cfg, classes, cuts, win_cls, first)
    ref = reference(cache, classes)
    ref["universe"] = densify(cache, K=256)
    F = len(acs.MAGNITUDE_FEATURES)
    n_cat, n_flag, n_bits = len(acs.CATEGORICAL_FIELDS), len(acs.FLAG_MASKS), len(acs.FLAG_BIT_NAMES)
    labels = classes
    M = len(labels)
    pairs = [(labels[i], labels[j], i, j) for i in range(M) for j in range(i + 1, M)]
    out = a.out; out.mkdir(parents=True, exist_ok=True)

    from ton_benign_audit import split_of  # noqa: F401  (kept for API parity)
    blocks_info = {}
    results = {}
    for bm in a.block_minutes:
        print(f"[boot] {bm}min x {a.boot} ...")
        t0 = time.time()
        arr, nb = run_length(cache, classes, ref, dist_pos, F, n_cat, n_flag, n_bits, bm, a.boot, a.seed)
        print(f"[boot] {bm}min done in {time.time()-t0:.1f}s")
        results[bm] = arr; blocks_info[bm] = nb

    for bm in a.block_minutes:
        arr = results[bm]
        ci_rows, rank_rows, cov_rows = [], [], []
        for mi, metric in enumerate(METRICS):
            sim = SIM[metric]
            pv = np.array([[arr[mi][b, x, y] for _, _, x, y in pairs] for b in range(arr.shape[1])])
            order = np.argsort(-pv if sim else pv, axis=1)
            ranks = np.empty_like(order); ranks[np.arange(arr.shape[1])[:, None], order] = np.arange(1, len(pairs) + 1)[None, :]
            for pi, (ca, cb, x, y) in enumerate(pairs):
                v = arr[mi][:, x, y]; v = v[np.isfinite(v)]
                lo, hi = (np.percentile(v, [2.5, 97.5]) if v.size else (np.nan, np.nan))
                ci_rows.append({"metric": metric, "class_a": ca, "class_b": cb,
                                "boot_mean": float(v.mean()), "boot_se": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                                "ci_low": float(lo), "ci_high": float(hi), "ci_width": float(hi - lo),
                                "n_replicates": int(v.size)})
                cov_rows.append({"metric": metric, "class_a": ca, "class_b": cb,
                                 "defined_fraction": float(np.isfinite(arr[mi][:, x, y]).mean())})
                ranks_i = ranks[:, pi]
                rank_rows.append({"metric": metric, "class_a": ca, "class_b": cb,
                                  "prob_top5": float((ranks_i <= 5).mean()), "rank_mean": float(ranks_i.mean()),
                                  "rank_median": float(np.median(ranks_i))})
        tag = f"{bm}min"
        pd.DataFrame(ci_rows).to_csv(out / f"bootstrap_pairwise_cis_{tag}.csv", index=False)
        pd.DataFrame(rank_rows).to_csv(out / f"bootstrap_rank_stability_{tag}.csv", index=False)
        pd.DataFrame(cov_rows).to_csv(out / f"bootstrap_defined_coverage_{tag}.csv", index=False)

    pd.DataFrame([{"class": c, "blocks_30min": blocks_info[30][c], "blocks_60min": blocks_info[60].get(c, 0)}
                  for c in classes]).to_csv(out / "bootstrap_class_blocks.csv", index=False)
    cfg_out = {"n_bootstrap": a.boot, "block_minutes": a.block_minutes, "primary_minutes": a.primary_minutes,
               "classes": labels, "sha256": sha, "cache_block_minutes": BLOCK_MIN,
               "resampling": "within each class's own contiguous training sequence; classes never concatenated",
               "normalisation": "training-only gmean/gstd recomputed per replicate",
               "wassertstein_subsample_cap": WASS_CAP, "blocks_per_class": blocks_info,
               "few_blocks_lt5": {str(bm): [c for c, n in blocks_info[bm].items() if n < 5]
                                  for bm in a.block_minutes}}
    (out / "bootstrap_config.json").write_text(json.dumps(cfg_out, indent=2, default=str), encoding="utf-8")

    L = ["# Step 1.4 - ToN-IoT temporal block bootstrap (within-class blocks)", "",
         f"- Blocks primary {a.primary_minutes} min, sensitivity {60} min; {a.boot} replicates.",
         "- Resampling is within each class's own sequence; classes never concatenated or shared.",
         "- Training-only normalisation recomputed per replicate.",
         f"- Data sha256 `{sha}`.",
         "", "## Classes with < 5 available blocks (cautious interpretation)",
         ""]
    for bm in a.block_minutes:
        few = [c for c, n in blocks_info[bm].items() if n < 5]
        L.append(f"- {bm} min: " + (", ".join(f"{c} ({blocks_info[bm][c]})" for c in few) or "none"))
    L += ["", "## Blocks per class", "",
          "| class | 30 min | 60 min |", "| --- | --- | --- |"]
    for c in classes:
        L.append(f"| {c} | {blocks_info[30][c]} | {blocks_info[60].get(c, 0)} |")
    L += ["", "## Notes", "",
          "- Rank stability and CI widths are reported per block length (separate CSVs).",
          "- Few blocks (e.g. injection at 60 min; mitm/ransomware) mean wide, cautious intervals;",
          "  500 replicates add Monte-Carlo precision only, not independent temporal observations.",
          "- ToN's disjoint attack periods (vs UNSW co-occurrence) are a dataset characteristic, not",
          "  a nuisance: resampling never mixes classes."]
    (out / "bootstrap_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[boot] wrote {out}")


if __name__ == "__main__":
    main()
