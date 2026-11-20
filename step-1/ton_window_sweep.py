"""Step 1.4 - NF-ToN-IoT-v3 window-size feasibility sweep (per-class contiguous split).

Runs on the canonical corrected split (benign window-matched). Class cuts are 60 s aligned and
every window size divides 60 s, so all five sizes stay isolated across train/val/test for every
class. Three views: natural, task-filtered (benign + designated class), and unlabeled context
(other attack flows, descriptive only). Benign is NOT downsampled.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

import attack_class_similarity as acs

SIZES = [1_000, 5_000, 10_000, 30_000, 60_000]
ALIGN_MS = 60_000
THRESH = [2, 5, 10, 20]
SPLITS = acs.SPLIT_NAMES


def load_cuts(manifest: Path):
    m = json.loads(manifest.read_text(encoding="utf-8"))
    return {c: (v["t70"], v["t85"]) for c, v in m["per_class_cuts"].items()}


def ts_range(data_path, cfg):
    lo, hi = None, None
    for chunk in acs._iter_csv(data_path, cfg, usecols=["FLOW_START_MILLISECONDS"]):
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        lo = int(t.min()) if lo is None else min(lo, int(t.min()))
        hi = int(t.max()) if hi is None else max(hi, int(t.max()))
    return lo, hi


def accumulate(data_path, cfg, sizes, bases, nws, classes):
    K = len(classes)
    total = {w: np.zeros(nws[w], dtype=np.int64) for w in sizes}
    benign = {w: np.zeros(nws[w], dtype=np.int64) for w in sizes}
    cls = {w: np.zeros((K, nws[w]), dtype=np.int64) for w in sizes}
    code = {c: i for i, c in enumerate(classes)}
    for chunk in acs._iter_csv(data_path, cfg, usecols=["FLOW_START_MILLISECONDS", "Attack"]):
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        a = chunk["Attack"].to_numpy()
        is_b = a == acs.BENIGN
        for w in sizes:
            rel = ((t - bases[w]) // w).astype(np.int64)
            total[w] += np.bincount(rel, minlength=nws[w])
            if is_b.any():
                benign[w] += np.bincount(rel[is_b], minlength=nws[w])
            for c, ci in code.items():
                m = a == c
                if m.any():
                    cls[w][ci] += np.bincount(rel[m], minlength=nws[w])
    return total, benign, cls


def _pct(n, d):
    return round(100.0 * n / d, 4) if d else 0.0


def analyse(w, base, total, benign, cls, classes, cuts, floors, thresh):
    nw = total.size
    wq_all = np.flatnonzero(total > 0)
    any_cls = cls.sum(axis=0)
    # headline per split (windows attributed by the class they contain)
    comp, nat, acls, filt, ratio, mix = [], [], [], [], [], []
    assigned = cls.argmax(axis=0)  # class index per window (classes disjoint)
    has = any_cls > 0
    for si, sname in enumerate(SPLITS):
        # split of an assigned window depends on its class
        ts = base + np.arange(nw) * w
        sp = np.full(nw, -1, dtype=np.int8)
        for ci, c in enumerate(classes):
            t70, t85 = cuts[c]
            m = cls[ci] > 0
            if m.any():
                s = np.where(ts[m] < t70, 0, np.where(ts[m] < t85, 1, 2))
                sp[np.flatnonzero(m)] = s
        mask = has & (sp == si)
        tot = total[mask]
        nwin = int(mask.sum())
        comp.append({"window_ms": w, "split": sname, "nonempty_windows": nwin,
                     "total_flows": int(tot.sum()), "median_total_flows": float(np.median(tot)) if nwin else 0.0,
                     "p90_total_flows": float(np.percentile(tot, 90)) if nwin else 0.0,
                     "max_total_flows": int(tot.max()) if nwin else 0})
        nd = np.zeros(nwin, dtype=int)
        for ci in range(len(classes)):
            nd += (cls[ci][mask] > 0)
        for k in range(0, len(classes) + 1):
            mix.append({"window_ms": w, "split": sname, "distinct_attack_classes": k,
                        "windows": int((nd == k).sum()), "window_pct": _pct((nd == k).sum(), nwin)})
    for ci, c in enumerate(classes):
        t70, t85 = cuts[c]
        wq = np.flatnonzero(cls[ci] > 0)
        ts = base + wq * w
        sp = np.where(ts < t70, 0, np.where(ts < t85, 1, 2))
        primary = floors[c]
        for si, sname in enumerate(SPLITS):
            sel = wq[sp == si]
            if sel.size == 0:
                continue
            ccf = cls[ci][sel]; tot = total[sel]; ben = benign[sel]
            att = tot - ben; other = att - ccf
            f = ben + ccf
            nat.append({"window_ms": w, "class": c, "is_primary": primary, "split": sname,
                        "windows": int(sel.size), "total_flows": int(tot.sum()),
                        "benign_flows": int(ben.sum()), "attack_flows": int(ccf.sum()),
                        "median_total_flows": float(np.median(tot)),
                        "p90_total_flows": float(np.percentile(tot, 90)),
                        "max_total_flows": int(tot.max()),
                        "pct_multi_attack_windows": 0.0, "pct_single_attack_windows": 100.0,
                        "median_benign_attack_ratio": float(np.median(ben / np.maximum(ccf, 1))),
                        "benign_attack_ratio_overall": (float(ben.sum() / ccf.sum()) if ccf.sum() else 0.0)})
            acls.append({"window_ms": w, "class": c, "is_primary": primary, "split": sname,
                         "nonempty_windows": int(sel.size),
                         "median_class_flows": float(np.median(ccf)),
                         "p90_class_flows": float(np.percentile(ccf, 90)),
                         "max_class_flows": int(ccf.max()), "class_flows": int(ccf.sum()),
                         **{f"windows_class_ge{t}": int((ccf >= t).sum()) for t in thresh}})
            filt.append({"window_ms": w, "class": c, "is_primary": primary, "split": sname,
                         "filtered_windows": int(sel.size),
                         "median_filtered_total": float(np.median(f)),
                         "p90_filtered_total": float(np.percentile(f, 90)),
                         "max_filtered_total": int(f.max()),
                         "pct_windows_class_lt2": _pct((ccf < 2).sum(), sel.size),
                         "pct_filtered_total_lt2": _pct((f < 2).sum(), sel.size),
                         "benign_attack_ratio_overall": (float(ben.sum() / ccf.sum()) if ccf.sum() else 0.0),
                         "median_benign_attack_ratio": float(np.median(ben / np.maximum(ccf, 1))),
                         "unlabeled_other_attack_flows": int(other.sum()),
                         "mean_other_attack_flows_per_window": float(other.mean()),
                         **{f"windows_filtered_total_ge{t}": int((f >= t).sum()) for t in thresh}})
            r = ben / np.maximum(ccf, 1)
            ratio.append({"window_ms": w, "class": c, "is_primary": primary, "split": sname,
                          "windows": int(sel.size), "ratio_median": float(np.median(r)),
                          "ratio_p25": float(np.percentile(r, 25)), "ratio_p75": float(np.percentile(r, 75)),
                          "ratio_p90": float(np.percentile(r, 90)), "ratio_max": float(r.max()),
                          "ratio_overall": (float(ben.sum() / ccf.sum()) if ccf.sum() else 0.0)})
    return comp, nat, acls, filt, ratio, mix


def figures(figdir, comp_df, ratio_df):
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in SPLITS:
        d = comp_df[comp_df.split == s].sort_values("window_ms")
        ax.plot(d.window_ms / 1000, d.median_total_flows, marker="o", label=s)
    ax.set(xlabel="window size (s)", ylabel="median flows / window", xscale="log",
           title="ToN window occupancy (per-class split)")
    ax.legend(); fig.tight_layout(); fig.savefig(figdir / "window_occupancy.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(comp_df[comp_df.split == "train"].sort_values("window_ms").window_ms / 1000,
            comp_df[comp_df.split == "train"].sort_values("window_ms").p90_total_flows, marker="o")
    ax.set(xlabel="window size (s)", ylabel="p90 flows / window (train)", xscale="log",
           title="Graph-window size distribution")
    fig.tight_layout(); fig.savefig(figdir / "graph_size_distributions.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in ratio_df["class"].unique():
        d = ratio_df[(ratio_df["class"] == c) & (ratio_df.split == "train")].sort_values("window_ms")
        ax.plot(d.window_ms / 1000, d.ratio_median, marker=".", label=c)
    ax.set(xlabel="window size (s)", ylabel="median benign:attack ratio", xscale="log", yscale="log",
           title="ToN benign:attack ratio (train)")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(figdir / "benign_attack_ratios.png"); plt.close(fig)


def report(path, comp_df, nat_df, acls_df, filt_df, classes, floors, ctx):
    sup = [c for c in classes if floors[c]]
    supp = [c for c in classes if not floors[c]]
    L = ["# Step 1.4 - NF-ToN-IoT-v3 window-size sweep (per-class contiguous split)", "",
         "*Canonical split: per-class contiguous, benign window-matched.*", "",
         f"- Dataset sha256 `{ctx['sha256']}`; rows {ctx['rows']:,}; 60 s-aligned class cuts.",
         f"- Primary classes (3-split floor): {', '.join(sup)}.",
         f"- Supplementary (train-supported, fail floor): {', '.join(supp)}.",
         f"- Thresholds for class flows: {THRESH}.",
         "- Mixing is 0 by construction: ToN attack classes occupy disjoint time windows, so no",
         "  window contains two attack classes. Benign:attack imbalance is left untouched.",
         "", "## Headline (train)", "",
         "| window | windows | median flows | p90 | max |", "| --- | --- | --- | --- | --- |"]
    for r in comp_df[comp_df.split == "train"].sort_values("window_ms").to_dict("records"):
        L.append(f"| {r['window_ms']//1000}s | {r['nonempty_windows']:,} | {r['median_total_flows']:.1f} | "
                 f"{r['p90_total_flows']:.1f} | {r['max_total_flows']:,} |")
    L += ["", "## Task-filtered feasibility (train) - windows with >= 5 class flows", "",
          "| window | class | primary | windows>=5 | >=10 | >=20 | median filtered total | median b:a ratio |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    d = filt_df[filt_df.split == "train"].merge(
        acls_df[acls_df.split == "train"][["window_ms", "class", "windows_class_ge5",
                                           "windows_class_ge10", "windows_class_ge20"]],
        on=["window_ms", "class"]).sort_values(["window_ms", "class"])
    for r in d.to_dict("records"):
        L.append(f"| {r['window_ms']//1000}s | {r['class']} | {r['is_primary']} | "
                 f"{r['windows_class_ge5']:,} | {r['windows_class_ge10']:,} | {r['windows_class_ge20']:,} | "
                 f"{r['median_filtered_total']:.1f} | {r['median_benign_attack_ratio']:.1f} |")
    L += ["", "## Notes", "",
          "- Unlabeled context is ~0 (classes disjoint), kept as a descriptive reference only.",
          "- No benign downsampling; the extreme per-class imbalance is preserved.",
          "- Window size is not auto-selected; review occupancy + task-filtered feasibility."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    repo = acs.find_repo_root(acs.STEP_DIR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(acs.DATASETS), default="ton")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--manifest", type=Path,
                    default=acs.STEP_DIR / "outputs" / "ton_class_contiguous_benignmatched" / "split_manifest.json")
    ap.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "ton_window_sweep")
    ap.add_argument("--sizes-ms", type=int, nargs="+", default=SIZES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--skip-hash", action="store_true")
    a = ap.parse_args()
    if a.data is None:
        a.data = repo / acs.DATASETS[a.dataset][0]
    cfg = {"chunk": a.chunk, "max_rows": a.max_rows}
    cuts = load_cuts(a.manifest)
    classes = list(cuts)
    fe = pd.read_csv(a.manifest.parent / "sample_feasibility.csv")
    floors = {r["class"]: bool(r["attack_floor_ok"]) for _, r in fe.iterrows()}

    sha = "skipped" if a.skip_hash else acs.sha256(a.data)
    lo, hi = ts_range(a.data, cfg)
    bases = {w: int(lo // w * w) for w in a.sizes_ms}
    nws = {w: int((hi - bases[w]) // w) + 1 for w in a.sizes_ms}
    print(f"[sweep] rows range {lo}..{hi}; nwindows@1s={nws[a.sizes_ms[0]]:,}")
    total, benign, cls = accumulate(a.data, cfg, a.sizes_ms, bases, nws, classes)
    print("[sweep] accumulated")

    comp, nat, acls, filt, ratio, mix = [], [], [], [], [], []
    for w in a.sizes_ms:
        c1, n1, a1, f1, r1, m1 = analyse(w, bases[w], total[w], benign[w], cls[w], classes, cuts, floors, THRESH)
        comp += c1; nat += n1; acls += a1; filt += f1; ratio += r1; mix += m1
    out = a.out; out.mkdir(parents=True, exist_ok=True)
    comp_df, nat_df, acls_df, filt_df, ratio_df, mix_df = (pd.DataFrame(x) for x in
                                                           (comp, nat, acls, filt, ratio, mix))
    (out / "window_size_comparison.csv").write_text(comp_df.to_csv(index=False), encoding="utf-8")
    nat_df.to_csv(out / "natural_window_statistics.csv", index=False)
    acls_df.to_csv(out / "attack_class_window_statistics.csv", index=False)
    filt_df.to_csv(out / "task_filtered_window_statistics.csv", index=False)
    ratio_df.to_csv(out / "benign_attack_ratio.csv", index=False)
    mix_df.to_csv(out / "window_mixing_statistics.csv", index=False)
    (out / "split_manifest.json").write_text(json.dumps({
        "mode": "per_class_contiguous_chronological (benign window-matched)",
        "align_ms": ALIGN_MS, "sizes_ms": a.sizes_ms,
        "per_class_cuts": {c: {"t70": cuts[c][0], "t85": cuts[c][1]} for c in classes},
        "per_size_base": {str(w): bases[w] for w in a.sizes_ms},
        "per_size_windows": {str(w): nws[w] for w in a.sizes_ms},
        "isolation": "class cuts are multiples of 60 s and every size divides 60 s, so no window "
                     "straddles a class split boundary",
    }, indent=2), encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps({
        "dataset": "NF-ToN-IoT-v3", "data_path": str(a.data.resolve()), "data_sha256": sha,
        "seed": a.seed, "boundary_policy": "per-class 60 s-aligned cuts", "thresholds": THRESH,
        "sizes_ms": a.sizes_ms, "rows": int(total[a.sizes_ms[0]].sum()),
        "primary_classes": [c for c in classes if floors[c]],
        "supplementary_classes": [c for c in classes if not floors[c]],
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")
    figures(out / "figures", comp_df, ratio_df)
    report(out / "window_sweep_report.md", comp_df, nat_df, acls_df, filt_df, classes, floors,
           {"sha256": sha, "rows": int(total[a.sizes_ms[0]].sum())})
    print(f"[sweep] wrote {out}")


if __name__ == "__main__":
    main()
