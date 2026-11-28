"""Step 1.2 window-size feasibility sweep (descriptive).

For window sizes 1/5/10/30/60 s (all divisors of the 60 s alignment), this measures
three views without changing the frozen similarity baseline:

* **natural**  - every flow retained (actual attack mixing),
* **task-filtered** - benign + one designated attack class (primary task graph),
* **unlabeled context** - extra attack flows the filtered graph would drop.

Split cuts come from :func:`attack_class_similarity.compute_global_split_cuts` aligned
to 60 s, so every window size divides the partitions exactly and no physical window
crosses a partition boundary.
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

WINDOW_SIZES_MS = [1_000, 5_000, 10_000, 30_000, 60_000]
ALIGN_MS = 60_000
THRESHOLDS = [2, 5, 10, 20, 50]
SPLITS = acs.SPLIT_NAMES


# --------------------------------------------------------------------------- #
# Core (testable) helpers
# --------------------------------------------------------------------------- #
def load_stream(data_path: Path, cfg: dict, attack_classes: list[str]):
    """One streaming pass -> (FLOW_START int64, class code int8, code_of map)."""
    code_of = {acs.BENIGN: 0}
    for i, a in enumerate(attack_classes, start=1):
        code_of[a] = i
    ts_parts, code_parts = [], []
    for chunk in acs._iter_csv(data_path, cfg, usecols=["FLOW_START_MILLISECONDS", "Attack"]):
        ts_parts.append(acs.validate_timestamp_column(
            chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
            "FLOW_START_MILLISECONDS"))
        code_parts.append(chunk["Attack"].map(code_of).to_numpy(dtype=np.int16))
    ts = np.concatenate(ts_parts)
    ccode = np.concatenate(code_parts)
    if np.isnan(ccode.astype(np.float64)).any():
        raise ValueError("unmapped Attack label encountered during streaming")
    return ts, ccode.astype(np.int16), code_of


def window_base(ts: np.ndarray, align_ms: int) -> int:
    """Origin for window ids: a multiple of ``align_ms``, so every divisor size aligns."""
    return int(ts.min() // align_ms * align_ms)


def split_of_windows(nw: int, w: int, t70: int, t85: int, base: int = 0) -> np.ndarray:
    """Split index per window id; exact because ``t70``/``t85`` are multiples of ``w``."""
    if (t70 - base) % w or (t85 - base) % w:
        raise ValueError(f"cut not aligned to window {w} (base {base})")
    c70, c85 = (t70 - base) // w, (t85 - base) // w
    ids = np.arange(nw)
    return np.where(ids < c70, 0, np.where(ids < c85, 1, 2)).astype(np.int8)


def window_counts(ts: np.ndarray, ccode: np.ndarray, K: int, w: int, base: int = 0):
    """Per-window total, benign, and per-attack-class counts (index 0 unused)."""
    wid = ((ts - base) // w).astype(np.int64)
    nw = int(wid.max()) + 1
    total = np.bincount(wid, minlength=nw)
    benign = np.bincount(wid[ccode == 0], minlength=nw)
    counts = [None] * (K + 1)
    for c in range(1, K + 1):
        counts[c] = np.bincount(wid[ccode == c], minlength=nw)
    return total, benign, counts


def _pct(num, den):
    return round(100.0 * num / den, 4) if den else 0.0


def analyse_size(w, ts, ccode, K, attack_classes, t70, t85, thresholds, base):
    total, benign, counts = window_counts(ts, ccode, K, w, base)
    split = split_of_windows(len(total), w, t70, t85, base)
    wsplit = split  # per window id

    natural_rows, mixing_rows, class_rows, filtered_rows, ratio_rows, comp_rows = [], [], [], [], [], []
    manifest = {}
    for si, sname in enumerate(SPLITS):
        m = (wsplit == si) & (total > 0)
        win_ids = np.nonzero(m)[0]
        tot = total[m]
        ben = benign[m]
        att = tot - ben
        nwin = int(m.sum())
        attack_total = int(att.sum())
        benign_total = int(ben.sum())

        ndist = np.zeros(nwin, dtype=np.int64)
        for c in range(1, K + 1):
            ndist += (counts[c][m] > 0)

        manifest[sname] = {"windows": nwin, "flows": int(tot.sum()),
                           "attack_flows": attack_total, "benign_flows": benign_total}

        natural_rows.append({
            "window_ms": w, "split": sname, "nonempty_windows": nwin,
            "total_flows": int(tot.sum()), "benign_flows": benign_total, "attack_flows": attack_total,
            "median_total_flows": float(np.median(tot)), "p90_total_flows": float(np.percentile(tot, 90)),
            "max_total_flows": int(tot.max()),
            "median_benign_flows": float(np.median(ben)), "median_attack_flows": float(np.median(att)),
            "pct_multi_attack_windows": _pct((ndist >= 2).sum(), nwin),
            "pct_single_attack_windows": _pct((ndist == 1).sum(), nwin),
            "pct_no_attack_windows": _pct((ndist == 0).sum(), nwin),
            "median_distinct_attack_classes": float(np.median(ndist)),
        })

        for k in range(0, K + 1):
            mixing_rows.append({"window_ms": w, "split": sname,
                                "distinct_attack_classes": k,
                                "windows": int((ndist == k).sum()),
                                "window_pct": _pct((ndist == k).sum(), nwin)})

        # benign:attack ratio over windows that contain attack traffic
        if attack_total:
            r = ben[att > 0] / att[att > 0]
            ratio_rows.append({
                "window_ms": w, "split": sname, "windows_with_attack": int((att > 0).sum()),
                "ratio_p25": float(np.percentile(r, 25)), "ratio_median": float(np.median(r)),
                "ratio_p75": float(np.percentile(r, 75)), "ratio_p90": float(np.percentile(r, 90)),
                "ratio_max": float(r.max()),
                "overall_benign": benign_total, "overall_attack": attack_total,
                "overall_ratio": float(benign_total / attack_total) if attack_total else 0.0,
            })

        n_ge = {}
        for c in range(1, K + 1):
            cc = counts[c][m]
            cw = cc[cc > 0]
            cls = attack_classes[c - 1]
            filtered_total = ben + cc
            other = att - cc
            ge = {f"windows_class_ge{t}": int((cc >= t).sum()) for t in thresholds}
            fge = {f"windows_filtered_total_ge{t}": int((filtered_total >= t).sum()) for t in thresholds}
            class_rows.append({
                "window_ms": w, "split": sname, "class": cls,
                "nonempty_windows": int(cw.size),
                "pct_windows_with_class": _pct(cw.size, nwin),
                "median_flows_per_class_window": float(np.median(cw)) if cw.size else 0.0,
                "p90_flows_per_class_window": float(np.percentile(cw, 90)) if cw.size else 0.0,
                "max_flows_per_class_window": int(cw.max()) if cw.size else 0,
                "class_flows": int(cc.sum()),
                "pct_of_attack_flows": _pct(cc.sum(), attack_total),
                **ge,
            })
            filtered_rows.append({
                "window_ms": w, "split": sname, "class": cls,
                "filtered_nonempty_windows": int((filtered_total > 0).sum()),
                "median_filtered_total": float(np.median(filtered_total)),
                "p90_filtered_total": float(np.percentile(filtered_total, 90)),
                "max_filtered_total": int(filtered_total.max()),
                "filtered_attack_flows": int(cc.sum()),
                "pct_of_attack_flows_retained": _pct(cc.sum(), attack_total),
                "mean_other_attack_flows_per_window": float(other.mean()) if nwin else 0.0,
                "other_attack_flows_dropped": int(other.sum()),
                "pct_windows_filtered_total_lt2": _pct((filtered_total < 2).sum(), nwin),
                "pct_windows_class_lt2": _pct((cc < 2).sum(), nwin),
                "task_benign_attack_ratio": (float(benign_total / cc.sum()) if cc.sum() else 0.0),
                "task_median_benign_attack_ratio": (
                    float(np.median(ben[cc > 0] / cc[cc > 0])) if (cc > 0).any() else 0.0),
                **fge,
            })
            n_ge[cls] = (int((cc >= 5).sum()), int((cc >= 10).sum()))

        comp_rows.append({
            "window_ms": w, "split": sname, "nonempty_windows": nwin,
            "total_flows": int(tot.sum()), "median_total_flows": float(np.median(tot)),
            "p90_total_flows": float(np.percentile(tot, 90)), "max_total_flows": int(tot.max()),
            "pct_multi_attack_windows": _pct((ndist >= 2).sum(), nwin),
            "pct_single_attack_windows": _pct((ndist == 1).sum(), nwin),
            "pct_no_attack_windows": _pct((ndist == 0).sum(), nwin),
            "classes_ge100_windows_ge5flows": sum(1 for v in n_ge.values() if v[0] >= 100),
            "classes_ge100_windows_ge10flows": sum(1 for v in n_ge.values() if v[1] >= 100),
            "median_benign_attack_ratio": (ratio_rows[-1]["ratio_median"] if attack_total else 0.0),
        })

    return dict(natural=natural_rows, mixing=mixing_rows, cls=class_rows,
                filtered=filtered_rows, ratio=ratio_rows, comp=comp_rows, manifest=manifest)


# --------------------------------------------------------------------------- #
# Figures + report
# --------------------------------------------------------------------------- #
def make_figures(fig_dir: Path, sizes, natural_df, mixing_df, ratio_df, windows_by_size, rng):
    fig_dir.mkdir(parents=True, exist_ok=True)
    xs = np.array(sizes) / 1000.0

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in SPLITS:
        d = natural_df[(natural_df["split"] == s)].sort_values("window_ms")
        ax.plot(d["window_ms"] / 1000.0, d["median_total_flows"], marker="o", label=s)
    ax.set(xlabel="window size (s)", ylabel="median flows / window", xscale="log",
           title="Window occupancy by split")
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "window_occupancy.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in SPLITS:
        d = natural_df[natural_df["split"] == s].sort_values("window_ms")
        ax.plot(d["window_ms"] / 1000.0, d["pct_multi_attack_windows"], marker="o",
                label=f"{s} multi")
    ax.set(xlabel="window size (s)", ylabel="% windows with >1 attack class", xscale="log",
           title="Attack-class mixing")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "attack_class_mixing.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [rng.choice(np.asarray(windows_by_size[w]), size=min(len(windows_by_size[w]), 50_000),
                       replace=False) for w in sizes]
    ax.boxplot(data, tick_labels=[f"{w//1000}s" for w in sizes], showfliers=False)
    ax.set(yscale="log", ylabel="flows / window (train)",
           title="Graph-window size distribution")
    fig.tight_layout(); fig.savefig(fig_dir / "graph_size_distributions.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in SPLITS:
        d = ratio_df[ratio_df["split"] == s].sort_values("window_ms")
        ax.plot(d["window_ms"] / 1000.0, d["ratio_median"], marker="o", label=s)
    ax.set(xlabel="window size (s)", ylabel="median benign:attack ratio", xscale="log",
           title="Benign-to-attack ratio (attack-bearing windows)")
    ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "benign_attack_ratios.png"); plt.close(fig)


def write_report(path: Path, sizes, manifest, natural_df, comp_df, filtered_df, cls_df, ctx):
    lines = [
        "# Step 1.2 -- Window-size feasibility sweep (NF-UNSW-NB15-v3)",
        "",
        "*Descriptive only; no similarity definition or baseline was changed.*",
        "",
        "## Boundary policy",
        "",
        f"- Split cuts are global chronological quantiles {ctx['split']} aligned to "
        f"**{ALIGN_MS // 1000} s**; every window size divides the alignment, so no physical "
        "window crosses a partition boundary.",
        f"- Seed {ctx['seed']}; dataset sha256 `{ctx['sha256']}`.",
        f"- Graph-flow thresholds: {THRESHOLDS}.",
        "",
        "## Actual split sizes (60 s aligned)",
        "",
        "| window | split | windows | flows | % flows |",
        "| --- | --- | --- | --- | --- |",
    ]
    for w in sizes:
        tot = sum(manifest[w][s]["flows"] for s in SPLITS)
        for s in SPLITS:
            r = manifest[w][s]
            lines.append(f"| {w // 1000}s | {s} | {r['windows']:,} | {r['flows']:,} | "
                         f"{100.0 * r['flows'] / tot:.3f} |")
    lines += ["", "## Headline comparison (train)", "",
              "| window | median flows | p90 | max | %multi-attack | %single-attack | "
              "classes w/ >=100 windows & >=5 flows |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for w in sizes:
        r = comp_df[(comp_df["window_ms"] == w) & (comp_df["split"] == "train")].iloc[0]
        lines.append(f"| {w // 1000}s | {r.median_total_flows:.1f} | {r.p90_total_flows:.1f} | "
                     f"{r.max_total_flows} | {r.pct_multi_attack_windows:.1f}% | "
                     f"{r.pct_single_attack_windows:.1f}% | {r.classes_ge100_windows_ge5flows} |")
    lines += ["", "## Task-filtered feasibility (train, benign + one class)", "",
              "| window | class | windows with >=5 class flows | >=10 | >=20 | "
              "median filtered total | % of attack flows retained |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    f = filtered_df[filtered_df["split"] == "train"].merge(
        cls_df[cls_df["split"] == "train"][
            ["window_ms", "class", "windows_class_ge5", "windows_class_ge10", "windows_class_ge20"]],
        on=["window_ms", "class"], how="left").sort_values(["window_ms", "class"])
    for r in f.to_dict("records"):
        lines.append(f"| {r['window_ms'] // 1000}s | {r['class']} | {r['windows_class_ge5']} | "
                     f"{r['windows_class_ge10']} | {r['windows_class_ge20']} | "
                     f"{r['median_filtered_total']:.1f} | {r['pct_of_attack_flows_retained']:.2f}% |")
    lines += ["", "## Notes",
              "",
              "- Mixing is measured on natural traffic; task-filtered graphs drop all attack "
              "classes outside the designated one (provisional primary policy).",
              "- Unlabeled-context traffic is reported as `other_attack_flows_dropped`; it is a "
              "future ablation, not part of the primary experiment.",
              "- Window size is **not** auto-selected. Choose from occupancy, mixing and "
              "task-filtered feasibility together."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    repo = acs.find_repo_root(acs.STEP_DIR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(acs.DATASETS), default=acs.DEFAULT_DATASET)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "unsw_window_sweep")
    ap.add_argument("--split", type=float, nargs=2, default=[0.70, 0.85])
    ap.add_argument("--window-sizes-ms", type=int, nargs="+", default=WINDOW_SIZES_MS)
    ap.add_argument("--align-ms", type=int, default=ALIGN_MS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--skip-hash", action="store_true")
    args = ap.parse_args()
    if args.data is None:
        args.data = repo / acs.DATASETS[args.dataset][0]
    args.dataset_name = acs.DATASETS.get(args.dataset, (None, Path(args.data).stem))[1]
    return args


def main():
    args = parse_args()
    out = args.out
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"split": tuple(args.split), "chunk": args.chunk, "max_rows": args.max_rows}
    for w in args.window_sizes_ms:
        if args.align_ms % w:
            raise SystemExit(f"window {w} does not divide align {args.align_ms}")

    print(f"[sweep] data   : {args.data.resolve()}")
    print(f"[sweep] outputs: {out.resolve()}")
    sha = "skipped" if args.skip_hash else acs.sha256(args.data)
    attack_classes = acs.discover_attack_classes(args.data, cfg)
    K = len(attack_classes)
    print(f"[sweep] {K} attack classes: {attack_classes}")
    t70, t85 = acs.compute_global_split_cuts(args.data, cfg, args.align_ms)
    print(f"[sweep] 60 s-aligned cuts: train<{t70}, val<{t85}")

    ts, ccode, _ = load_stream(args.data, cfg, attack_classes)
    print(f"[sweep] streamed {ts.size:,} rows")

    rng = np.random.default_rng(args.seed)
    base = window_base(ts, args.align_ms)
    print(f"[sweep] window-id base: {base}")
    natural, mixing, cls, filtered, ratio, comp, manifest = [], [], [], [], [], [], {}
    windows_by_size = {}
    for w in args.window_sizes_ms:
        res = analyse_size(w, ts, ccode, K, attack_classes, t70, t85, THRESHOLDS, base)
        natural += res["natural"]; mixing += res["mixing"]; cls += res["cls"]
        filtered += res["filtered"]; ratio += res["ratio"]; comp += res["comp"]
        manifest[w] = res["manifest"]
        total, _, _ = window_counts(ts, ccode, K, w, base)
        split = split_of_windows(len(total), w, t70, t85, base)
        windows_by_size[w] = total[(split == 0) & (total > 0)]
        print(f"[sweep] {w // 1000}s done: {manifest[w]['train']['windows']:,} train windows")

    natural_df = pd.DataFrame(natural)
    mixing_df = pd.DataFrame(mixing)
    cls_df = pd.DataFrame(cls)
    filtered_df = pd.DataFrame(filtered)
    ratio_df = pd.DataFrame(ratio)
    comp_df = pd.DataFrame(comp)

    natural_df.to_csv(out / "natural_window_statistics.csv", index=False)
    mixing_df.to_csv(out / "window_mixing_statistics.csv", index=False)
    cls_df.to_csv(out / "attack_class_window_statistics.csv", index=False)
    filtered_df.to_csv(out / "task_filtered_window_statistics.csv", index=False)
    ratio_df.to_csv(out / "benign_attack_ratio.csv", index=False)
    comp_df.to_csv(out / "window_size_comparison.csv", index=False)

    split_manifest = {
        "align_ms": args.align_ms, "cuts": {"t70": int(t70), "t85": int(t85)},
        "split_quantiles": list(args.split), "window_sizes_ms": args.window_sizes_ms,
        "per_window": {str(w): manifest[w] for w in args.window_sizes_ms},
    }
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    run_manifest = {
        "dataset": args.dataset_name, "data_path": str(args.data.resolve()), "data_sha256": sha,
        "seed": args.seed, "boundary_policy": f"global chronological quantiles aligned to {args.align_ms} ms",
        "thresholds": THRESHOLDS, "window_sizes_ms": args.window_sizes_ms,
        "n_rows": int(ts.size), "attack_classes": attack_classes,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    make_figures(fig_dir, args.window_sizes_ms, natural_df, mixing_df, ratio_df, windows_by_size, rng)
    write_report(out / "window_sweep_report.md", args.window_sizes_ms, manifest,
                 natural_df, comp_df, filtered_df, cls_df,
                 {"split": list(args.split), "seed": args.seed, "sha256": sha})
    print(f"[sweep] wrote {out}")


if __name__ == "__main__":
    main()
