"""Step 1.3 - uncertainty (temporal block bootstrap) and temporal relation structure.

Two read-only analyses over the frozen split (train-only for similarity):

``bootstrap``        - temporal block bootstrap CIs for the pairwise similarity
                       matrices. Blocks are contiguous time spans inside the training
                       partition; all classes move together inside a block; the same
                       blocks are used for every pair; training-only standardisation is
                       recomputed inside every replicate.
``relation-windows`` - per-window relation statistics at 10 s and 5 s, task-filtered
                       (benign + the designated class), reporting distributions and
                       temporal variability rather than a single aggregate.

Neither analysis changes the frozen baseline in ``outputs/unsw_full``.
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
SIMILARITY = {"cosine": True, "wasserstein": False, "jensen_shannon": False,
              "relation_profile": True, "flag_bit_similarity": True}
REL_FIELDS = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO"]


# --------------------------------------------------------------------------- #
# Shared: block cache (one streaming pass over the training partition)
# --------------------------------------------------------------------------- #
def _new_class_block(F, n_cat, n_flag, n_bits):
    return {"c_valid": np.zeros(F, dtype=np.int64), "c_sum": np.zeros(F),
            "c_sumsq": np.zeros(F), "dist": defaultdict(list),
            "cat": [Counter() for _ in range(n_cat)],
            "src": Counter(), "dst": Counter(), "pair": Counter(),
            "flagbit": np.zeros(n_bits, dtype=np.int64),
            "flag_valid": np.zeros(n_flag, dtype=np.int64)}


def _groups(keys: np.ndarray, idx: np.ndarray):
    """Yield (unique_key, row_indices) using a single sort (vectorised grouping)."""
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    uniq, starts = np.unique(ks, return_index=True)
    ends = np.append(starts[1:], ks.size)
    for i, k in enumerate(uniq):
        yield int(k), idx[order[starts[i]:ends[i]]]


def load_blocks(data_path: Path, cfg: dict, attack_classes: list[str], cuts, cut_ms=FINE_MS):
    """Cache per-(60 s block, class) sufficient statistics for the training partition."""
    features = acs.MAGNITUDE_FEATURES
    F = len(features)
    dist_pos = [features.index(f) for f in acs.DIST_FEATURES]
    all_classes = [acs.BENIGN] + list(attack_classes)
    code_of = {n: i for i, n in enumerate(all_classes)}
    n_cat, n_flag, n_bits = len(acs.CATEGORICAL_FIELDS), len(acs.FLAG_MASKS), len(acs.FLAG_BIT_NAMES)
    t70, t85 = cuts

    G_valid: dict[int, np.ndarray] = {}
    G_sum: dict[int, np.ndarray] = {}
    G_sumsq: dict[int, np.ndarray] = {}
    CB: dict[tuple, dict] = {}
    train_blocks = set()

    for chunk in acs._iter_csv(data_path, cfg):
        codes = chunk["Attack"].map(code_of).to_numpy(dtype=np.int16)
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        split = np.where(t < t70, 0, np.where(t < t85, 1, 2)).astype(np.int8)
        tr = split == 0
        if not tr.any():
            continue
        bid = (t // cut_ms).astype(np.int64)
        X = chunk[features].to_numpy(dtype=np.float64)
        valid = np.isfinite(X) & (X >= 0.0)
        L = np.log1p(np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None))
        L[~valid] = np.nan
        Ltr, bid_tr = L[tr], bid[tr]
        train_blocks.update(np.unique(bid_tr).tolist())

        bs, inv = np.unique(bid_tr, return_inverse=True)
        for p in range(F):
            col = Ltr[:, p]
            fin = np.isfinite(col)
            w = np.where(fin, col, 0.0)
            if bs.size:
                sv = np.bincount(inv, weights=fin.astype(float), minlength=bs.size)
                sm = np.bincount(inv, weights=w, minlength=bs.size)
                sq = np.bincount(inv, weights=w * w, minlength=bs.size)
            else:
                sv = sm = sq = np.zeros(0)
            for i, b in enumerate(bs):
                b = int(b)
                if b not in G_valid:
                    G_valid[b] = np.zeros(F, dtype=np.int64)
                    G_sum[b] = np.zeros(F)
                    G_sumsq[b] = np.zeros(F)
                G_valid[b][p] += int(sv[i]); G_sum[b][p] += sm[i]; G_sumsq[b][p] += sq[i]

        ai = np.flatnonzero(tr & (codes > 0))
        if ai.size:
            key = bid[ai].astype(np.int64) * 64 + codes[ai].astype(np.int64)
            src = chunk[acs.HOST_SRC].to_numpy(); dst = chunk[acs.HOST_DST].to_numpy()
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
            for kk, rows in _groups(key, ai):
                ci = int(kk % 64); b = int(kk // 64)
                cb = CB.get((b, ci))
                if cb is None:
                    cb = CB[(b, ci)] = _new_class_block(F, n_cat, n_flag, n_bits)
                Lt = L[rows]
                fin = np.isfinite(Lt)
                cb["c_valid"] += fin.sum(axis=0)
                cb["c_sum"] += np.nansum(Lt, axis=0)
                cb["c_sumsq"] += np.nansum(Lt * Lt, axis=0)
                for fpos in dist_pos:
                    vals = Lt[:, fpos][fin[:, fpos]]
                    if vals.size:
                        cb["dist"][fpos].append(vals)
                for j, field in enumerate(acs.CATEGORICAL_FIELDS):
                    ok = cat_ok[field][rows]
                    if ok.any():
                        cb["cat"][j].update(cat_codes[field][rows][ok].tolist())
                cb["src"].update(src[rows].tolist())
                cb["dst"].update(dst[rows].tolist())
                cb["pair"].update(zip(src[rows].tolist(), dst[rows].tolist()))
                cb["flag_valid"] += fval[rows].sum(axis=0)
                cb["flagbit"] += bitmat[rows].sum(axis=0)

    # full-train reference structures (used as the FIXED padding/universe)
    full_src = [Counter() for _ in range(len(all_classes))]
    full_dst = [Counter() for _ in range(len(all_classes))]
    full_pair = [Counter() for _ in range(len(all_classes))]
    full_cat = [[Counter() for _ in acs.CATEGORICAL_FIELDS] for _ in range(len(all_classes))]
    full_cvalid = np.zeros((len(all_classes), F), dtype=np.int64)
    for (b, ci), cb in CB.items():
        full_src[ci].update(cb["src"]); full_dst[ci].update(cb["dst"]); full_pair[ci].update(cb["pair"])
        full_cvalid[ci] += cb["c_valid"]
        for j in range(n_cat):
            full_cat[ci][j].update(cb["cat"][j])
    L_host = max([max(len(full_src[c]), len(full_dst[c])) for c in range(len(all_classes))] or [1])
    L_pair = max([len(full_pair[c]) for c in range(len(all_classes))] or [1])
    universe = {}
    for j, field in enumerate(acs.CATEGORICAL_FIELDS):
        glob = Counter()
        for c in range(len(all_classes)):
            glob.update(full_cat[c][j])
        universe[field] = list(glob.keys())
    consistent = [p for p in range(F) if int(full_cvalid[1:, p].min()) >= cfg["min_feature_samples"]]

    return {"cut_ms": cut_ms, "train_blocks": sorted(train_blocks), "G": (G_valid, G_sum, G_sumsq),
            "CB": CB, "attack_classes": attack_classes, "all_classes": all_classes,
            "F": F, "dist_pos": dist_pos, "L_host": L_host, "L_pair": L_pair,
            "universe": universe, "consistent": consistent,
            "cat_idx": {f: acs.CATEGORICAL_FIELDS.index(f) for f in acs.CATEGORICAL_FIELDS},
            "n_cat": n_cat, "n_flag": n_flag, "n_bits": n_bits}


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _merge_class(cache, sel_blocks, ci):
    F = cache["F"]
    cv = np.zeros(F, dtype=np.int64); cs = np.zeros(F); cq = np.zeros(F)
    dist = defaultdict(list)
    cat = [Counter() for _ in range(cache["n_cat"])]
    src, dst, pair = Counter(), Counter(), Counter()
    flagbit = np.zeros(cache["n_bits"], dtype=np.int64)
    flag_valid = np.zeros(cache["n_flag"], dtype=np.int64)
    for b in sel_blocks:
        cb = cache["CB"].get((b, ci))
        if cb is None:
            continue
        cv += cb["c_valid"]; cs += cb["c_sum"]; cq += cb["c_sumsq"]
        for fpos, arrs in cb["dist"].items():
            dist[fpos].extend(arrs)
        for j in range(cache["n_cat"]):
            cat[j].update(cb["cat"][j])
        src.update(cb["src"]); dst.update(cb["dst"]); pair.update(cb["pair"])
        flagbit += cb["flagbit"]; flag_valid += cb["flag_valid"]
    return {"cv": cv, "cs": cs, "cq": cq, "dist": dist, "cat": cat,
            "src": src, "dst": dst, "pair": pair, "flagbit": flagbit, "flag_valid": flag_valid}


def _global_g(cache, sel_blocks):
    F = cache["F"]
    Gv, Gs, Gq = cache["G"]
    gv = np.zeros(F, dtype=np.int64); gs = np.zeros(F); gq = np.zeros(F)
    for b in sel_blocks:
        if b in Gv:
            gv += Gv[b]; gs += Gs[b]; gq += Gq[b]
    gmean = np.divide(gs, gv, out=np.zeros(F), where=gv > 0)
    gvar = np.divide(gq, gv, out=np.zeros(F), where=gv > 0) - gmean ** 2
    gstd = np.sqrt(np.clip(gvar, 0.0, None)); gstd[gstd == 0] = 1.0
    return gv, gmean, gstd


def replicate(cache, sel_blocks, rng, wass_cap=3000):
    names = cache["all_classes"]
    M = len(cache["attack_classes"])
    m_idx = [names.index(c) for c in cache["attack_classes"]]
    gv, gmean, gstd = _global_g(cache, sel_blocks)
    classes = {c: _merge_class(cache, sel_blocks, names.index(c)) for c in cache["attack_classes"]}

    cmean = np.vstack([
        np.divide(classes[c]["cs"], classes[c]["cv"], out=np.full(cache["F"], np.nan),
                  where=classes[c]["cv"] > 0) for c in cache["attack_classes"]])
    z = (cmean - gmean) / gstd
    consistent = [p for p in cache["consistent"] if np.all(np.isfinite(z[:, p]))]
    cos = acs.cosine_matrix(z[:, consistent]) if len(consistent) >= 2 else np.zeros((M, M))

    wnum = np.zeros((M, M)); wden = np.zeros((M, M), dtype=np.int64)
    for k, fpos in enumerate(cache["dist_pos"]):
        smp = []
        for c in cache["attack_classes"]:
            vals = classes[c]["dist"].get(fpos, [])
            a = np.concatenate(vals) if vals else np.zeros(0)
            z2 = (a - gmean[fpos]) / gstd[fpos]
            smp.append(z2)
        for a in range(M):
            for b in range(a + 1, M):
                sa, sb = smp[a], smp[b]
                if sa.size >= 2 and sb.size >= 2:
                    if sa.size > wass_cap:
                        sa = rng.choice(sa, wass_cap, replace=False)
                    if sb.size > wass_cap:
                        sb = rng.choice(sb, wass_cap, replace=False)
                    w = float(wasserstein_distance(sa, sb))
                    wnum[a, b] += w; wnum[b, a] += w; wden[a, b] += 1; wden[b, a] += 1
    wass = np.divide(wnum, wden, out=np.zeros((M, M)), where=wden > 0)

    cat_items = []
    for j, field in enumerate(acs.CATEGORICAL_FIELDS):
        vecs = [acs._prob_vector(classes[c]["cat"][j], cache["universe"][field]) for c in cache["attack_classes"]]
        cat_items.append((field, vecs, [v.sum() > 0 for v in vecs]))
    js, _, _ = acs.aggregate_pairwise_js(cat_items, M, cache["attack_classes"])

    def blk(cls_rec, block):
        if block == "source_host_degree":
            return acs.sorted_share_distribution(cls_rec["src"], cache["L_host"])
        if block == "dest_host_degree":
            return acs.sorted_share_distribution(cls_rec["dst"], cache["L_host"])
        if block == "host_pair_degree":
            return acs.sorted_share_distribution(cls_rec["pair"], cache["L_pair"])
        f = block.split("flow_to_")[1]
        field = {"src_port": "L4_SRC_PORT", "dst_port": "L4_DST_PORT",
                 "protocol": "PROTOCOL", "service": "L7_PROTO"}[f]
        return acs._prob_vector(cls_rec["cat"][cache["cat_idx"][field]], cache["universe"][field])

    rel_items = []
    for b in acs.RELATION_BLOCKS:
        vecs = [blk(classes[c], b) for c in cache["attack_classes"]]
        rel_items.append((b, vecs, [v.sum() > 0 for v in vecs]))
    struct, _, _ = acs.aggregate_pairwise_js(rel_items, M, cache["attack_classes"])
    rel = np.clip(1.0 - struct, 0.0, 1.0)

    prev = np.vstack([
        acs.flag_bit_prevalence(classes[c]["flagbit"][None, :],
                                classes[c]["flag_valid"][None, :])[0][0]
        for c in cache["attack_classes"]])
    flags = acs.cosine_matrix(prev)

    return {"cosine": cos, "wasserstein": wass, "jensen_shannon": js,
            "relation_profile": rel, "flag_bit_similarity": flags}


def _pairs(M, labels):
    out = []
    for a in range(M):
        for b in range(a + 1, M):
            out.append((labels[a], labels[b], a, b))
    return out


def _rank_stats(arr, pairs, sim):
    """Rank of each pair across replicates (1 = most similar)."""
    B = arr.shape[0]
    P = len(pairs)
    pv = np.empty((B, P))
    for i, (_, _, a, b) in enumerate(pairs):
        pv[:, i] = arr[:, a, b]
    order = np.argsort(-pv if sim else pv, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(B)[:, None], order] = np.arange(1, P + 1)[None, :]
    return pv, ranks


def run_bootstrap(cache, cfg, out_dir: Path, lengths_ms, primary_ms, n_boot, n_sens, seed, acf_lags=30):
    names = cache["all_classes"]
    train_flows = {}
    for c in cache["attack_classes"]:
        ci = names.index(c)
        train_flows[c] = sum(sum(cb["pair"].values())
                             for b in cache["train_blocks"] if (cb := cache["CB"].get((b, ci))))
    labels = [c for c in cache["attack_classes"] if train_flows[c] >= cfg["split_floor"]]
    if len(labels) < 2:
        raise SystemExit("fewer than 2 attack classes meet the train support floor")
    cache["attack_classes"] = labels
    FC = cache["F"]
    full_cv = np.zeros((len(labels), FC), dtype=np.int64)
    for i, c in enumerate(labels):
        ci = names.index(c)
        for b in cache["train_blocks"]:
            cb = cache["CB"].get((b, ci))
            if cb is not None:
                full_cv[i] += cb["c_valid"]
    cache["consistent"] = [p for p in range(FC)
                           if int(full_cv[:, p].min()) >= cfg["min_feature_samples"]]
    M = len(labels)
    pairs = _pairs(M, labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    fine = cache["cut_ms"]
    mult = {L: L // fine for L in lengths_ms}
    if any(L % fine for L in lengths_ms):
        raise SystemExit("block lengths must be multiples of the 60 s alignment")

    def blocks_for(L):
        m = mult[L]
        agg = defaultdict(list)
        for b in cache["train_blocks"]:
            agg[b // m].append(b)
        return [v for _, v in sorted(agg.items())]

    coarse_counts = {L: len(blocks_for(L)) for L in lengths_ms}

    # temporal dependence diagnostic: lag-1..10 autocorrelation of per-fine-block class flows
    ac_rows = []
    counts = {c: Counter() for c in cache["all_classes"]}
    for (b, ci), cb in cache["CB"].items():
        counts[cache["all_classes"][ci]][b] += int(sum(cb["pair"].values()))
    for c in labels:
        series = pd.Series(counts[c]).sort_index()
        x = series.to_numpy(dtype=float)
        if x.std() > 0 and x.size > 3:
            for lag in range(1, acf_lags + 1):
                if x.size - lag >= 3:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r = float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
                    ac_rows.append({"class": c, "lag_blocks": lag, "autocorr": r})
    pd.DataFrame(ac_rows).to_csv(out_dir / "bootstrap_autocorrelation.csv", index=False)
    # decay summary: first lag where |autocorr| < 0.2
    decay = []
    acdf = pd.DataFrame(ac_rows)
    for c in labels:
        d = acdf[(acdf["class"] == c) & (acdf["autocorr"].abs() < 0.2)]
        decay.append({"class": c, "first_lag_below_0.2": int(d["lag_blocks"].min()) if len(d) else None,
                      "lag1": float(acdf[acdf["class"] == c]["autocorr"].iloc[0])})
    pd.DataFrame(decay).to_csv(out_dir / "bootstrap_autocorrelation_decay.csv", index=False)

    rng = np.random.default_rng(seed)

    def run_length(L, B):
        blks = blocks_for(L)
        nb = len(blks)
        store = {m: [] for m in METRICS}
        for _ in range(B):
            sel = []
            for i in rng.integers(0, nb, size=nb):
                sel.extend(blks[i])
            mats = replicate(cache, sel, rng)
            for m in METRICS:
                store[m].append(mats[m])
        return np.array([store[m] for m in METRICS])  # (metrics, B, M, M)

    print(f"[boot] primary {primary_ms//1000}s x {n_boot} replicates ...")
    t0 = time.time()
    prim = run_length(primary_ms, n_boot)
    print(f"[boot] primary done in {time.time()-t0:.1f}s")
    sens = {}
    for L in lengths_ms:
        if L == primary_ms:
            continue
        print(f"[boot] sensitivity {L//1000}s x {n_sens} ...")
        sens[L] = run_length(L, n_sens)

    base = pd.read_csv(cfg["baseline_dir"] / "pair_comparison.csv")

    def observed(metric, ca, cb):
        r = base[((base.class_a == ca) & (base.class_b == cb)) |
                 ((base.class_a == cb) & (base.class_b == ca))]
        col = {"cosine": "cosine", "wasserstein": "wasserstein", "jensen_shannon": "js_distance",
               "relation_profile": "relation_similarity", "flag_bit_similarity": "flag_bits_similarity"}[metric]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    ci_rows, rank_rows, cov_rows = [], [], []
    prim_rank = {}
    for mi, metric in enumerate(METRICS):
        sim = SIMILARITY[metric]
        _, ranks = _rank_stats(prim[mi], pairs, sim)
        prim_rank[metric] = ranks
        for pi, (ca, cb, a, b) in enumerate(pairs):
            vals = prim[mi][:, a, b]
            defined = np.isfinite(vals)
            v = vals[defined]
            obs = observed(metric, ca, cb)
            lo, hi = (np.percentile(v, [2.5, 97.5]) if v.size else (np.nan, np.nan))
            ci_rows.append({"metric": metric, "class_a": ca, "class_b": cb, "observed": obs,
                            "boot_mean": float(v.mean()) if v.size else np.nan,
                            "boot_se": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                            "ci_low": float(lo), "ci_high": float(hi),
                            "ci_width": float(hi - lo) if v.size else np.nan,
                            "n_replicates": int(v.size)})
            cov_rows.append({"metric": metric, "class_a": ca, "class_b": cb,
                             "defined_fraction": float(defined.mean()), "n_replicates": int(vals.size)})
            rr = ranks[:, pi]
            rank_rows.append({"metric": metric, "class_a": ca, "class_b": cb,
                              "prob_top5": float((rr <= 5).mean()), "rank_mean": float(rr.mean()),
                              "rank_median": float(np.median(rr)), "rank_min": int(rr.min()),
                              "rank_max": int(rr.max()), "n_pairs": len(pairs)})

    pd.DataFrame(ci_rows).to_csv(out_dir / "bootstrap_pairwise_cis.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(out_dir / "bootstrap_rank_stability.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(out_dir / "bootstrap_defined_coverage.csv", index=False)

    # rank stability compared across block lengths
    sens_rank = {}
    for L, arr in sens.items():
        sens_rank[L] = {metric: _rank_stats(arr[mi], pairs, SIMILARITY[metric])[1]
                        for mi, metric in enumerate(METRICS)}
    cmp_rows = []
    for metric in METRICS:
        pr = prim_rank[metric]
        for pi, (ca, cb, a, b) in enumerate(pairs):
            row = {"metric": metric, "class_a": ca, "class_b": cb,
                   f"prob_top5_{primary_ms // 1000}s": float((pr[:, pi] <= 5).mean()),
                   f"rank_mean_{primary_ms // 1000}s": float(pr[:, pi].mean())}
            for L in sorted(sens):
                r = sens_rank[L][metric][:, pi]
                row[f"prob_top5_{L // 1000}s"] = float((r <= 5).mean())
                row[f"rank_mean_{L // 1000}s"] = float(r.mean())
            cmp_rows.append(row)
    pd.DataFrame(cmp_rows).to_csv(out_dir / "bootstrap_rank_stability_comparison.csv", index=False)

    # sensitivity: mean CI width per metric per length
    sens_rows = []
    for mi, metric in enumerate(METRICS):
        sens_rows.append({"metric": metric, "block_ms": primary_ms,
                          "mean_ci_width": float(np.nanmean(
                              np.abs(np.percentile(prim[mi][:, [p[2] for p in pairs], [p[3] for p in pairs]],
                                                   97.5, axis=0)
                                     - np.percentile(prim[mi][:, [p[2] for p in pairs], [p[3] for p in pairs]],
                                                     2.5, axis=0))))})
        for L, arr in sens.items():
            aa = arr[mi]
            widths = []
            for _, _, a, b in pairs:
                v = aa[:, a, b]; v = v[np.isfinite(v)]
                if v.size:
                    widths.append(np.percentile(v, 97.5) - np.percentile(v, 2.5))
            sens_rows.append({"metric": metric, "block_ms": L, "mean_ci_width": float(np.mean(widths))})
    pd.DataFrame(sens_rows).to_csv(out_dir / "bootstrap_block_length_sensitivity.csv", index=False)

    cfg_out = {"n_bootstrap": n_boot, "n_sensitivity": n_sens, "seed": seed,
               "fine_block_ms": fine, "block_lengths_ms": lengths_ms, "primary_block_ms": primary_ms,
               "acf_lags": acf_lags,
               "coarse_blocks_per_length_ms": {str(L): n for L, n in coarse_counts.items()},
               "primary_temporal_blocks": coarse_counts[primary_ms],
               "metrics": METRICS, "normalisation": "training-only gmean/gstd recomputed per replicate",
               "fixed_universe": True, "fixed_host_padding": {"L_host": cache["L_host"], "L_pair": cache["L_pair"]},
               "wassertstein_subsample_cap": 3000, "train_blocks": len(cache["train_blocks"]),
               "n_pairs": len(pairs), "classes": labels}
    (out_dir / "bootstrap_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")

    lines = ["# Step 1.3 - temporal block bootstrap", "",
             f"- Primary block length {primary_ms//1000}s, {n_boot} replicates; seed {seed}.",
             f"- Training fine (60 s) blocks: {len(cache['train_blocks'])}; "
             f"coarse blocks: " + ", ".join(f"{L//1000}s={n}" for L, n in sorted(coarse_counts.items()))
             + ".",
             f"- **Caveat:** coarse-block count limits effective temporal information "
             f"(primary = {coarse_counts[primary_ms]} blocks); replicate count improves Monte-Carlo "
             "precision only, not the number of independent blocks.",
             f"- Classes: {', '.join(labels)}.",
             "- Blocks are contiguous training time spans; all classes (incl. benign) move together;",
             "  the same blocks are used for every pair; gmean/gstd are recomputed per replicate.",
             "- Categorical universe and host-degree padding are FIXED to the full-train values.",
             "",
             "## Defined coverage (primary)", "",
             f"- All pairs defined in {min(r['defined_fraction'] for r in cov_rows)*100:.1f}%-"
             f"{max(r['defined_fraction'] for r in cov_rows)*100:.1f}% of replicates.",
             "",
             "## 95% CI width by block length (mean over pairs)", "",
             "| metric | " + " | ".join(f"{L//1000}s" for L in [primary_ms] + sorted(sens)) + " |",
             "| --- | " + " | ".join("---" for _ in [primary_ms] + sorted(sens)) + " |"]
    for metric in METRICS:
        row = [f"{metric}"]
        for L in [primary_ms] + sorted(sens):
            r = pd.DataFrame(sens_rows).query("metric == @metric and block_ms == @L")
            row.append(f"{r['mean_ci_width'].iloc[0]:.4f}" if len(r) else "")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Top pairs (primary block length)", ""]
    for metric in METRICS:
        d = pd.DataFrame(ci_rows).query("metric == @metric").copy()
        sim = SIMILARITY[metric]
        d = d.sort_values("observed", ascending=not sim).head(5)
        lines.append(f"**{metric}**")
        for r in d.itertuples():
            lines.append(f"- {r.class_a}-{r.class_b}: observed {r.observed:.4f}, "
                         f"95% CI [{r.ci_low:.4f}, {r.ci_high:.4f}]")
        lines.append("")
    lines += ["", "## Rank stability across block lengths (probability of top-5)", "",
              "| metric | pair | " + " | ".join(f"{L//1000}s" for L in [primary_ms] + sorted(sens)) + " |",
              "| --- | --- | " + " | ".join("---" for _ in [primary_ms] + sorted(sens)) + " |"]
    cdf = pd.DataFrame(cmp_rows)
    for metric in METRICS:
        d = cdf[cdf["metric"] == metric].sort_values(f"rank_mean_{primary_ms // 1000}s").head(6)
        for r in d.to_dict("records"):
            cells = [f"{r[f'prob_top5_{primary_ms // 1000}s']:.2f}"]
            cells += [f"{r[f'prob_top5_{L // 1000}s']:.2f}" for L in sorted(sens)]
            lines.append(f"| {metric} | {r['class_a']}-{r['class_b']} | " + " | ".join(cells) + " |")
    (out_dir / "bootstrap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"primary": prim, "sens": sens}


# --------------------------------------------------------------------------- #
# Per-window relation statistics
# --------------------------------------------------------------------------- #
def _entropy(counter: Counter):
    tot = sum(counter.values())
    if tot <= 0:
        return 0.0
    p = np.array(list(counter.values()), dtype=float) / tot
    h = -(p * np.log2(p)).sum()
    k = (p > 0).sum()
    return float(h / np.log2(k)) if k > 1 else 0.0


def _window_metrics(class_src, class_dst, class_pair, ports):
    n_pair = sum(class_pair.values())
    top_pair = (max(class_pair.values()) / n_pair) if n_pair else 0.0
    return {"n_src_hosts": len(class_src), "n_dst_hosts": len(class_dst),
            "n_host_pairs": n_pair, "host_pair_top_share": top_pair,
            "n_src_ports": len(ports["L4_SRC_PORT"]), "n_dst_ports": len(ports["L4_DST_PORT"]),
            "n_protocols": len(ports["PROTOCOL"]), "n_services": len(ports["L7_PROTO"]),
            "dst_port_entropy": _entropy(ports["L4_DST_PORT"]),
            "src_port_entropy": _entropy(ports["L4_SRC_PORT"]),
            "protocol_entropy": _entropy(ports["PROTOCOL"]),
            "service_entropy": _entropy(ports["L7_PROTO"])}


def relation_windows(data_path, cfg, attack_classes, cuts, out_dir: Path, sizes_ms):
    all_classes = [acs.BENIGN] + list(attack_classes)
    code_of = {n: i for i, n in enumerate(all_classes)}
    t70, t85 = cuts
    base = t70 - (t70 % max(sizes_ms))
    cols = ["FLOW_START_MILLISECONDS", "Attack", acs.HOST_SRC, acs.HOST_DST] + REL_FIELDS
    # per (size, split, window, class): counters
    store = {w: {} for w in sizes_ms}
    benign = {w: Counter() for w in sizes_ms}
    for chunk in acs._iter_csv(data_path, cfg, usecols=cols):
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        codes = chunk["Attack"].map(code_of).to_numpy(dtype=np.int16)
        split = np.where(t < t70, 0, np.where(t < t85, 1, 2)).astype(np.int8)
        src = chunk[acs.HOST_SRC].to_numpy(); dst = chunk[acs.HOST_DST].to_numpy()
        pv = {f: chunk[f].to_numpy(dtype=np.float64) for f in REL_FIELDS}
        for w in sizes_ms:
            wid = ((t - base) // w).astype(np.int64)
            bidx = np.flatnonzero(codes == 0)
            if bidx.size:
                for kk, rows in _groups(wid[bidx] * 3 + split[bidx], bidx):
                    benign[w][(int(kk // 3), int(kk % 3))] += int(rows.size)
            ai = np.flatnonzero(codes > 0)
            key = wid[ai].astype(np.int64) * 64 + codes[ai].astype(np.int64)
            for kk, rows in _groups(key, ai):
                ci = int(kk % 64); b = int(kk // 64)
                sp = int(split[rows[0]])
                rec = store[w].setdefault((b, sp, ci), {"src": Counter(), "dst": Counter(),
                                                        "pair": Counter(),
                                                        "ports": {f: Counter() for f in REL_FIELDS}})
                rec["src"].update(src[rows].tolist()); rec["dst"].update(dst[rows].tolist())
                rec["pair"].update(zip(src[rows].tolist(), dst[rows].tolist()))
                for f in REL_FIELDS:
                    v = pv[f][rows]
                    ok = np.isfinite(v) & (v >= 0) & (v == np.floor(v))
                    if ok.any():
                        rec["ports"][f].update(v[ok].astype(np.int64).tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    for w in sizes_ms:
        rows = []
        for (b, sp, ci), rec in store[w].items():
            m = _window_metrics(rec["src"], rec["dst"], rec["pair"], rec["ports"])
            rows.append({"window_ms": w, "split": acs.SPLIT_NAMES[sp], "window_id": b,
                         "class": all_classes[ci], "n_class_flows": sum(rec["pair"].values()),
                         "n_benign_flows": benign[w].get((b, sp), 0), **m})
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"relation_window_statistics_{w//1000}s.csv", index=False)

    dist_rows, stab_rows = [], []
    metrics = ["n_src_hosts", "n_dst_hosts", "n_host_pairs", "host_pair_top_share",
               "n_src_ports", "n_dst_ports", "n_protocols", "n_services",
               "dst_port_entropy", "src_port_entropy", "protocol_entropy", "service_entropy"]
    for w in sizes_ms:
        df = pd.read_csv(out_dir / f"relation_window_statistics_{w//1000}s.csv")
        for (sp, cls), g in df.groupby(["split", "class"]):
            for met in metrics + ["n_class_flows", "n_benign_flows"]:
                v = g[met].to_numpy(dtype=float)
                dist_rows.append({"window_ms": w, "split": sp, "class": cls, "metric": met,
                                  "n_windows": v.size, "median": float(np.median(v)),
                                  "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                                  "p90": float(np.percentile(v, 90)), "mean": float(v.mean()),
                                  "std": float(v.std(ddof=1)) if v.size > 1 else 0.0})
            tot = len(df[(df.split == sp)])
            stab = {"window_ms": w, "split": sp, "class": cls, "windows_present": len(g),
                    "pct_windows_with_class": 100.0 * len(g) / tot if tot else 0.0}
            for met in metrics + ["n_class_flows"]:
                v = g[met].to_numpy(dtype=float)
                stab[f"{met}_cv"] = float(v.std(ddof=1) / v.mean()) if v.size > 1 and v.mean() else 0.0
            stab_rows.append(stab)
    pd.DataFrame(dist_rows).to_csv(out_dir / "relation_temporal_distributions.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(out_dir / "relation_temporal_stability.csv", index=False)

    lines = ["# Step 1.3 - per-window relation statistics (task-filtered)", "",
             "Per window, statistics are computed on the designated class's flows; the window must contain",
             "the class (windows where it is absent are excluded). `n_benign_flows` is the benign traffic in",
             "the same window under the task-filtered policy. Windows are reported at 10 s (primary) and 5 s.",
             ""]
    for w in sizes_ms:
        d = pd.DataFrame(dist_rows)
        d = d[(d.window_ms == w) & (d.split == "train") & (d.metric == "dst_port_entropy")]
        lines += [f"## {w//1000}s - train dst-port entropy (median per class)", "",
                  "| class | windows | median | IQR | p90 |", "| --- | --- | --- | --- | --- |"]
        for r in d.to_dict("records"):
            lines.append(f"| {r['class']} | {r['n_windows']} | {r['median']:.3f} | "
                         f"{r['q75'] - r['q25']:.3f} | {r['p90']:.3f} |")
        lines.append("")
    (out_dir / "temporal_relation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    repo = acs.find_repo_root(acs.STEP_DIR)
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap")
    b.add_argument("--dataset", choices=sorted(acs.DATASETS), default="unsw")
    b.add_argument("--data", type=Path, default=None)
    b.add_argument("--baseline-dir", type=Path, default=acs.STEP_DIR / "outputs" / "unsw_full")
    b.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "unsw_uncertainty" / "bootstrap")
    b.add_argument("--block-lengths-s", type=int, nargs="+", default=[60, 300, 900])
    b.add_argument("--primary-length-s", type=int, default=900)
    b.add_argument("--boot", type=int, default=500)
    b.add_argument("--sens-boot", type=int, default=500)
    b.add_argument("--acf-lags", type=int, default=30)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--chunk", type=int, default=250_000)
    b.add_argument("--max-rows", type=int, default=None)

    r = sub.add_parser("relation-windows")
    r.add_argument("--dataset", choices=sorted(acs.DATASETS), default="unsw")
    r.add_argument("--data", type=Path, default=None)
    r.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "unsw_uncertainty" / "temporal_relations")
    r.add_argument("--window-sizes-ms", type=int, nargs="+", default=[10_000, 5_000])
    r.add_argument("--chunk", type=int, default=250_000)
    r.add_argument("--max-rows", type=int, default=None)

    args = ap.parse_args()
    if args.data is None:
        args.data = repo / acs.DATASETS[args.dataset][0]
    cfg = {"chunk": args.chunk, "max_rows": args.max_rows, "min_feature_samples": 2,
           "split": (0.70, 0.85), "split_floor": 1000}
    cuts = acs.compute_global_split_cuts(args.data, cfg, FINE_MS)
    attack_classes = acs.discover_attack_classes(args.data, cfg)
    print(f"[1.3] {len(attack_classes)} classes; 60s-aligned cuts {cuts}")

    if args.cmd == "bootstrap":
        cache = load_blocks(args.data, cfg, attack_classes, cuts)
        cfg["baseline_dir"] = args.baseline_dir
        run_bootstrap(cache, cfg, args.out, [L * 1000 for L in args.block_lengths_s],
                      args.primary_length_s * 1000, args.boot, args.sens_boot, args.seed,
                      acf_lags=args.acf_lags)
        print(f"[1.3] wrote {args.out}")
    else:
        relation_windows(args.data, cfg, attack_classes, cuts, args.out, args.window_sizes_ms)
        print(f"[1.3] wrote {args.out}")


if __name__ == "__main__":
    main()
