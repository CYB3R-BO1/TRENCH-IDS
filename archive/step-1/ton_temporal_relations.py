"""Step 1.5 - per-window relation statistics for NF-ToN-IoT-v3 (canonical benign-matched split).

Task-filtered condition: benign + the designated attack class (other attack classes excluded).
Two populations per class and split:
  A = windows containing >=1 flow of the class (plus matched benign)
  B = windows containing benign but no flow of the class (same temporal interval)

The CSV is time-sorted, so windows are finalised incrementally with bounded memory.
Metrics are computed on the task-filtered graph (graph_*) and, for population A, on the
class's own flows (attack_*). Undefined (no flows) is NaN, distinct from genuine 0 entropy.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

import attack_class_similarity as acs

SPLITS = acs.SPLIT_NAMES
FIELDS = [("src", "host"), ("dst", "host"), ("pair", "host"),
          ("src_port", "L4_SRC_PORT"), ("dst_port", "L4_DST_PORT"),
          ("protocol", "PROTOCOL"), ("service", "L7_PROTO")]


def load_cfg(manifest: Path, coverage: Path):
    m = json.loads(manifest.read_text(encoding="utf-8"))
    cuts = {c: (v["t70"], v["t85"]) for c, v in m["per_class_cuts"].items()}
    cov = pd.read_csv(coverage)
    bounds = {}
    for r in cov.to_dict("records"):
        c = r["class"]
        first = int(pd.Timestamp(r["first_occurrence"]).value // 10 ** 6)
        last = int(pd.Timestamp(r["last_occurrence"]).value // 10 ** 6)
        bounds[c] = (first, last)
    return cuts, bounds


def new_acc():
    return {"benign": {k: Counter() for k, _ in FIELDS}, "attack": {k: Counter() for k, _ in FIELDS},
            "nb": 0, "na": 0, "cls": None}


def add(acc, side, vals, hist):
    for (k, col), values in zip(FIELDS, vals):
        if values:
            acc[side][k].update(values)
            hist[k] += len(values)


def metrics(counters, n_flows):
    out = {"n_flows": int(n_flows)}
    for k, _ in FIELDS:
        cnt = counters[k]
        tot = sum(cnt.values())
        if tot == 0:
            out[f"{k}_distinct"] = 0
            out[f"{k}_top_share"] = np.nan
            out[f"{k}_entropy"] = np.nan
            continue
        kk = len(cnt)
        out[f"{k}_distinct"] = kk
        out[f"{k}_top_share"] = max(cnt.values()) / tot
        if kk > 1:
            p = np.fromiter((v / tot for v in cnt.values()), dtype=float)
            out[f"{k}_entropy"] = float(-(p * np.log2(p)).sum() / np.log2(kk))
        else:
            out[f"{k}_entropy"] = 0.0
    return out


def split_of(ts, cut):
    return 0 if ts < cut[0] else (1 if ts < cut[1] else 2)


def finalise(acc, w, wid, cuts, bounds, rows, empty_marks):
    ts = wid * w
    cls = acc["cls"]
    if cls is None:
        # population B: assign to the class whose temporal interval contains this window
        owner = None
        for c, (first, last) in bounds.items():
            lo = (first // w) * w
            hi = (last // w) * w + w
            if lo <= ts < hi:
                owner = c
                break
        if owner is None:
            return
        cls = owner
        pop = "B"
    else:
        pop = "A"
    cut = cuts[cls]
    sp = split_of(ts, cut)
    nb, na = acc["nb"], acc["na"]
    if nb + na == 0:
        return
    gcounts = {k: acc["benign"][k] + acc["attack"][k] for k, _ in FIELDS}
    gm = metrics(gcounts, nb + na)
    row = {"window_ms": w, "split": SPLITS[sp], "class": cls, "window_id": int(wid),
           "population": pop, "n_benign": nb, "n_attack": na}
    row.update({f"graph_{k}": v for k, v in gm.items()})
    if pop == "A":
        am = metrics(acc["attack"], na)
        row.update({f"attack_{k}": v for k, v in am.items()})
    else:
        row.update({f"attack_{k}": np.nan for k in
                    [x for k, _ in FIELDS for x in (f"{k}_distinct", f"{k}_top_share", f"{k}_entropy")]}
                    | {"attack_n_flows": np.nan})
    rows.append(row)


def main():
    repo = acs.find_repo_root(acs.STEP_DIR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(acs.DATASETS), default="ton")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--manifest", type=Path,
                    default=acs.STEP_DIR / "outputs" / "ton_class_contiguous_benignmatched" / "split_manifest.json")
    ap.add_argument("--coverage", type=Path,
                    default=acs.STEP_DIR / "outputs" / "ton_class_contiguous" / "class_temporal_coverage.csv")
    ap.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "ton_uncertainty" / "temporal_relations")
    ap.add_argument("--sizes-ms", type=int, nargs="+", default=[10_000, 5_000])
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--skip-hash", action="store_true")
    a = ap.parse_args()
    if a.data is None:
        a.data = repo / acs.DATASETS[a.dataset][0]
    cfg = {"chunk": a.chunk, "max_rows": a.max_rows}
    cuts, bounds = load_cfg(a.manifest, a.coverage)
    classes = list(cuts)
    code = {c: i + 1 for i, c in enumerate(classes)}
    sha = "skipped" if a.skip_hash else acs.sha256(a.data)

    pending = {w: {} for w in a.sizes_ms}   # wid -> acc
    last_wid = {w: None for w in a.sizes_ms}
    rows, hist_all = [], {w: {k: 0 for k, _ in FIELDS} for w in a.sizes_ms}
    observed = defaultdict(int)             # (w, c, split) -> windows with flows
    host_id = {}

    def hid(h):
        i = host_id.get(h)
        if i is None:
            i = len(host_id); host_id[h] = i
        return i

    n_rows = 0
    for chunk in acs._iter_csv(a.data, cfg):
        n_rows += len(chunk)
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        atk = chunk["Attack"].to_numpy()
        codes = chunk["Attack"].map(code).fillna(0).to_numpy(dtype=np.int16)
        src = chunk[acs.HOST_SRC].to_numpy(); dst = chunk[acs.HOST_DST].to_numpy()
        sip = np.fromiter((hid(h) for h in src), dtype=np.int64, count=src.size)
        dip = np.fromiter((hid(h) for h in dst), dtype=np.int64, count=dst.size)
        pair = sip * (1 << 21) + dip
        raw = {"src": sip, "dst": dip, "pair": pair,
               "src_port": chunk["L4_SRC_PORT"].to_numpy(dtype=np.int64),
               "dst_port": chunk["L4_DST_PORT"].to_numpy(dtype=np.int64),
               "protocol": chunk["PROTOCOL"].to_numpy(dtype=np.int64),
               "service": chunk["L7_PROTO"].to_numpy(dtype=np.int64)}
        for w in a.sizes_ms:
            wid_all = (t // w).astype(np.int64)
            order = np.argsort(wid_all, kind="stable")
            ws = wid_all[order]
            uniq, st = np.unique(ws, return_index=True)
            en = np.append(st[1:], ws.size)
            for i, wid in enumerate(uniq):
                idx = order[st[i]:en[i]]
                acc = pending[w].get(int(wid))
                if acc is None:
                    acc = pending[w][int(wid)] = new_acc()
                bm = codes[idx] == 0
                if bm.any():
                    bi = idx[bm]
                    acc["nb"] += int(bi.size)
                    add(acc, "benign", [raw[k][bi].tolist() for k, _ in FIELDS], hist_all[w])
                am = codes[idx] > 0
                if am.any():
                    ai = idx[am]
                    cs = np.unique(codes[ai])
                    if acc["cls"] is None:
                        acc["cls"] = classes[int(cs[0]) - 1]
                    acc["na"] += int(ai.size)
                    add(acc, "attack", [raw[k][ai].tolist() for k, _ in FIELDS], hist_all[w])
            # finalise every window strictly before the last one in this chunk
            if uniq.size:
                cutoff = int(uniq[-1])
                done = [k for k in pending[w] if k < cutoff]
                for k in done:
                    acc = pending[w].pop(k)
                    clsx = acc["cls"]
                    if clsx is None:
                        pass
                    # observed counting needs the owner class even for B
                    ts = k * w
                    owner = clsx
                    if owner is None:
                        for c, (first, last) in bounds.items():
                            lo = (first // w) * w; hi = (last // w) * w + w
                            if lo <= ts < hi:
                                owner = c; break
                    if owner is not None:
                        observed[(w, owner, split_of(ts, cuts[owner]))] += 1
                    finalise(acc, w, k, cuts, bounds, rows, None)
                # keep the last window open
        if n_rows % 3_000_000 < cfg["chunk"]:
            print(f"  scanned {n_rows:,} rows ...", flush=True)
    # flush remaining
    for w in a.sizes_ms:
        for k in sorted(pending[w]):
            acc = pending[w].pop(k)
            ts = k * w
            owner = acc["cls"]
            if owner is None:
                for c, (first, last) in bounds.items():
                    lo = (first // w) * w; hi = (last // w) * w + w
                    if lo <= ts < hi:
                        owner = c; break
            if owner is not None:
                observed[(w, owner, split_of(ts, cuts[owner]))] += 1
            finalise(acc, w, k, cuts, bounds, rows, None)
    print(f"[rel] done: {n_rows:,} rows, {len(rows):,} window rows", flush=True)

    out = a.out; (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for w in a.sizes_ms:
        d = df[df.window_ms == w]
        d.to_csv(out / f"relation_window_statistics_{w//1000}s.csv", index=False)

    # distributions (graph_* and attack_*, by population)
    metrics_list = [f"{pre}{k}_{suf}" for pre in ("graph_", "attack_")
                    for k, _ in FIELDS for suf in ("distinct", "top_share", "entropy")] + \
                   [f"{pre}n_flows" for pre in ("graph_", "attack_")]
    dist_rows, stab_rows = [], []
    for w in a.sizes_ms:
        d = df[df.window_ms == w]
        for (sp, cls, pop), g in d.groupby(["split", "class", "population"]):
            for m in metrics_list:
                v = g[m].to_numpy(dtype=float)
                defined = v[np.isfinite(v)]
                dist_rows.append({"window_ms": w, "split": sp, "class": cls, "population": pop,
                                  "metric": m, "n_windows": int(v.size),
                                  "n_defined": int(defined.size), "n_undefined": int(v.size - defined.size),
                                  "median": float(np.median(defined)) if defined.size else np.nan,
                                  "q25": float(np.percentile(defined, 25)) if defined.size else np.nan,
                                  "q75": float(np.percentile(defined, 75)) if defined.size else np.nan,
                                  "p90": float(np.percentile(defined, 90)) if defined.size else np.nan,
                                  "mean": float(defined.mean()) if defined.size else np.nan,
                                  "std": float(defined.std(ddof=1)) if defined.size > 1 else np.nan,
                                  "cv": (float(defined.std(ddof=1) / defined.mean())
                                         if defined.size > 1 and defined.mean() else np.nan)})
            stab = {"window_ms": w, "split": sp, "class": cls, "population": pop,
                    "windows": int(len(g))}
            for k, _ in FIELDS:
                for suf in ("top_share", "entropy", "distinct"):
                    col = f"graph_{k}_{suf}"
                    v = g[col].to_numpy(dtype=float); v = v[np.isfinite(v)]
                    stab[f"graph_{k}_{suf}_cv"] = float(v.std(ddof=1) / v.mean()) if v.size > 1 and v.mean() else np.nan
            stab_rows.append(stab)
    pd.DataFrame(dist_rows).to_csv(out / "relation_temporal_distributions.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(out / "relation_temporal_stability.csv", index=False)

    # present vs absent (graph metrics)
    pva = []
    for w in a.sizes_ms:
        d = df[df.window_ms == w]
        for (sp, cls), g in d.groupby(["split", "class"]):
            A = g[g.population == "A"]; B = g[g.population == "B"]
            for m in metrics_list:
                if not m.startswith("graph_"):
                    continue
                va = A[m].to_numpy(dtype=float); va = va[np.isfinite(va)]
                vb = B[m].to_numpy(dtype=float); vb = vb[np.isfinite(vb)]
                pva.append({"window_ms": w, "split": sp, "class": cls, "metric": m,
                            "n_A": int(va.size), "n_B": int(vb.size),
                            "median_A": float(np.median(va)) if va.size else np.nan,
                            "median_B": float(np.median(vb)) if vb.size else np.nan,
                            "p90_A": float(np.percentile(va, 90)) if va.size else np.nan,
                            "p90_B": float(np.percentile(vb, 90)) if vb.size else np.nan,
                            "delta_median": (float(np.median(va) - np.median(vb))
                                             if va.size and vb.size else np.nan)})
    pd.DataFrame(pva).to_csv(out / "relation_present_vs_absent.csv", index=False)

    # figures: key graph metrics by class (population A) at 10s and 5s
    key = ["graph_dst_port_entropy", "graph_pair_top_share", "graph_src_port_distinct", "graph_service_entropy"]
    for w in a.sizes_ms:
        d = df[(df.window_ms == w) & (df.population == "A") & (df.split == "train")]
        fig, axes = plt.subplots(1, len(key), figsize=(16, 4))
        for ax, m in zip(axes, key):
            data = [d[d["class"] == c][m].dropna().to_numpy() for c in classes]
            ax.boxplot(data, tick_labels=classes, showfliers=False)
            ax.set_title(m.replace("graph_", "")); ax.tick_params(axis="x", rotation=90, labelsize=7)
        fig.tight_layout(); fig.savefig(out / "figures" / f"relation_{w//1000}s.png"); plt.close(fig)

    # report
    L = ["# Step 1.5 - ToN per-window relation statistics (task-filtered)", "",
         f"- Canonical benign-matched split; windows {', '.join(str(w//1000)+'s' for w in a.sizes_ms)}.",
         "- Graph = benign + designated class; other attack classes excluded.",
         "- Population A = class present; B = benign-only windows in the same class interval.",
         f"- Data sha256 `{sha}`.", "",
         "## Window counts", "",
         "| window | split | class | A | B |", "| --- | --- | --- | --- | --- |"]
    for w in a.sizes_ms:
        d = df[df.window_ms == w]
        for (sp, cls), g in d.groupby(["split", "class"]):
            L.append(f"| {w//1000}s | {sp} | {cls} | {int((g.population=='A').sum())} | {int((g.population=='B').sum())} |")
    L += ["", "## 5 s vs 10 s, Backdoor / Injection / Password / XSS (train, population A, graph)", "",
          "| class | window | dst-port entropy | pair top-share | src-port distinct | service entropy |",
          "| --- | --- | --- | --- | --- | --- |"]
    for cls in ["Backdoor", "injection", "password", "xss"]:
        for w in a.sizes_ms:
            d = df[(df.window_ms == w) & (df.split == "train") & (df["class"] == cls) & (df.population == "A")]
            if d.empty:
                continue
            L.append(f"| {cls} | {w//1000}s | {d.graph_dst_port_entropy.median():.3f} | "
                     f"{d.graph_pair_top_share.median():.3f} | {d.graph_src_port_distinct.median():.1f} | "
                     f"{d.graph_service_entropy.median():.3f} |")
    L += ["", "## Notes", "",
          "- Undefined (no flows) is NaN and counted separately from genuine 0 entropy.",
          "- Empty graphs are excluded from distributions; their counts are derivable from the interval",
          "  length minus observed A+B windows.",
          "- Similarity of aggregate profiles does not imply per-window similarity; compare the",
          "  distributions and the present-vs-absent deltas."]
    (out / "temporal_relation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[rel] wrote {out}")


if __name__ == "__main__":
    main()
