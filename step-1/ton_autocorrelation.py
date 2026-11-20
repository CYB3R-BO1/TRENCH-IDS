"""Step 1.4 - per-class temporal autocorrelation for NF-ToN-IoT-v3 (60 s bins).

Each attack class has its own contiguous training interval, so ACF is computed per class on its
own 60 s flow-count series (zero bins included, periods outside the class interval excluded).
Classes are NEVER concatenated. Includes a detrending and a log1p sensitivity check.
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

BIN_MS = 60_000
LAG_CAP = 180
BLOCK_CANDIDATES_S = [300, 600, 900, 1800, 3600]


def load(manifest: Path):
    m = json.loads(manifest.read_text(encoding="utf-8"))
    return {c: (v["t70"], v["t85"]) for c, v in m["per_class_cuts"].items()}


def bin_counts(data_path, cfg, cuts, classes):
    cnt = {c: Counter() for c in classes}
    first = {}
    for chunk in acs._iter_csv(data_path, cfg, usecols=["FLOW_START_MILLISECONDS", "Attack"]):
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        a = chunk["Attack"].to_numpy()
        b = (t // BIN_MS).astype(np.int64)
        for c in classes:
            t70 = cuts[c][0]
            m = (a == c) & (t < t70)
            if m.any():
                u, inv = np.unique(b[m], return_inverse=True)
                cc = np.bincount(inv, minlength=u.size)
                for w, n in zip(u.tolist(), cc.tolist()):
                    cnt[c][int(w)] += int(n)
                first[c] = min(first.get(c, 10 ** 18), int(u[0]))
    return cnt, first


def _acf(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)
    L = min(max_lag, x.size - 2)
    if denom == 0 or L < 1:
        return np.full(max_lag, np.nan), 0
    out = np.empty(L)
    for k in range(1, L + 1):
        out[k - 1] = float(np.sum(x[:-k] * x[k:]) / denom)
    return out, L


def _detrend(x):
    t = np.arange(x.size, dtype=float)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def _decay(r):
    """First lag |r|<0.2 and first sustained crossing (all later lags <0.2)."""
    below = np.abs(r) < 0.2
    first = int(np.argmax(below) + 1) if below.any() else None
    sust = None
    for k in range(r.size):
        if below[k:].all():
            sust = k + 1
            break
    return first, sust


def main():
    repo = acs.find_repo_root(acs.STEP_DIR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(acs.DATASETS), default="ton")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--manifest", type=Path,
                    default=acs.STEP_DIR / "outputs" / "ton_class_contiguous_benignmatched" / "split_manifest.json")
    ap.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "ton_uncertainty" / "temporal")
    ap.add_argument("--lags", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--skip-hash", action="store_true")
    a = ap.parse_args()
    if a.data is None:
        a.data = repo / acs.DATASETS[a.dataset][0]
    cfg = {"chunk": a.chunk, "max_rows": a.max_rows}
    cuts = load(a.manifest)
    classes = list(cuts)
    fe = pd.read_csv(a.manifest.parent / "sample_feasibility.csv")
    floors = {r["class"]: bool(r["attack_floor_ok"]) for _, r in fe.iterrows()}
    sha = "skipped" if a.skip_hash else acs.sha256(a.data)
    print(f"[acf] data {a.data.resolve()}")
    cnt, first = bin_counts(a.data, cfg, cuts, classes)

    rows, decay_rows, sens_rows, dep_rows = [], [], [], []
    series = {}
    for c in classes:
        t70 = cuts[c][0]
        fb = min(cnt[c]) if cnt[c] else t70 // BIN_MS
        lb = t70 // BIN_MS
        n = max(lb - fb, 2)
        x = np.zeros(n, dtype=float)
        for w, v in cnt[c].items():
            if fb <= w < lb:
                x[w - fb] = v
        series[c] = (x, fb)
        raw, L = _acf(x, a.lags)
        if L == a.lags and np.abs(raw[-1]) >= 0.2:
            raw, L = _acf(x, min(LAG_CAP, x.size - 2))
        det, _ = _acf(_detrend(x), len(raw))
        log, _ = _acf(np.log1p(x), len(raw))
        f_raw, s_raw = _decay(raw)
        f_det, s_det = _decay(det)
        f_log, s_log = _decay(log)
        for k in range(raw.size):
            rows.append({"class": c, "lag": k + 1, "acf_raw": raw[k],
                         "acf_detrended": det[k] if k < det.size else np.nan,
                         "acf_log1p": log[k] if k < log.size else np.nan})
        interval_s = (lb - fb) * 60
        dep_rows.append({
            "class": c, "is_primary": floors[c],
            "train_start": pd.to_datetime(fb * BIN_MS, unit="ms", utc=True),
            "train_end": pd.to_datetime(lb * BIN_MS, unit="ms", utc=True),
            "n_bins": int(n), "total_train_flows": int(x.sum()), "bin_mean": float(x.mean()),
            "bin_std": float(x.std()), "lag1_acf": float(raw[0]) if raw.size else np.nan,
            "first_below_0.2_raw": f_raw, "sustained_below_0.2_raw": s_raw,
            "first_below_0.2_detrended": f_det, "sustained_below_0.2_detrended": s_det,
            "acf_lags_computed": int(raw.size),
            **{f"blocks_at_{b}s": int(interval_s // b) for b in BLOCK_CANDIDATES_S},
        })
        decay_rows.append({"class": c, "is_primary": floors[c], "lag1": float(raw[0]) if raw.size else np.nan,
                           "first_below_0.2": f_raw, "sustained_below_0.2": s_raw,
                           "sustained_minutes": (s_raw or 0)})
        sens_rows.append({"class": c, "is_primary": floors[c],
                          "raw_first": f_raw, "raw_sustained": s_raw,
                          "detrended_first": f_det, "detrended_sustained": s_det,
                          "log1p_first": f_log, "log1p_sustained": s_log})

    out = a.out; (out / "figures").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_class_autocorrelation.csv", index=False)
    pd.DataFrame(decay_rows).to_csv(out / "autocorrelation_decay.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(out / "autocorrelation_sensitivity.csv", index=False)
    pd.DataFrame(dep_rows).to_csv(out / "class_temporal_dependence.csv", index=False)

    # suggested conservative block = max sustained crossing among primary classes, rounded up
    prim = [r for r in decay_rows if r["is_primary"] and r["sustained_below_0.2"]]
    suggested_min = max((r["sustained_minutes"] for r in prim), default=0)
    suggested_s = max(60, int(np.ceil(suggested_min * 60 / 60.0) * 60))

    prim_c = [c for c in classes if floors[c]]
    supp_c = [c for c in classes if not floors[c]]
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in prim_c:
        d = pd.DataFrame(rows); d = d[d["class"] == c]
        ax.plot(d["lag"], d["acf_raw"], marker=".", label=c)
    ax.axhline(0.2, color="k", ls="--", lw=0.8); ax.axhline(-0.2, color="k", ls="--", lw=0.8)
    ax.set(xlabel="lag (60 s bins)", ylabel="ACF", title="ToN primary classes - raw ACF")
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(out / "figures" / "acf_primary.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in supp_c:
        d = pd.DataFrame(rows); d = d[d["class"] == c]
        ax.plot(d["lag"], d["acf_raw"], marker=".", label=c)
    ax.axhline(0.2, color="k", ls="--", lw=0.8)
    ax.set(xlabel="lag (60 s bins)", ylabel="ACF", title="ToN supplementary classes - raw ACF")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "figures" / "acf_supplementary.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    dd = pd.DataFrame(decay_rows).sort_values("sustained_below_0.2")
    ax.barh(dd["class"], dd["sustained_below_0.2"], color=["C0" if p else "C1" for p in dd["is_primary"]])
    ax.set(xlabel="sustained decay lag (x 60 s)", title="First sustained |ACF| < 0.2")
    fig.tight_layout(); fig.savefig(out / "figures" / "decay_lags.png"); plt.close(fig)

    L = ["# ToN-IoT per-class temporal autocorrelation (60 s bins)", "",
         "## Configuration", "",
         f"- Resolution {BIN_MS//1000}s; per class, its own training interval (zero bins included,",
         "  outside-period excluded); classes never concatenated.",
         f"- Lags: >= {a.lags}, extended to {LAG_CAP} if not decayed; detrending + log1p sensitivity.",
         f"- Data sha256 `{sha}`.",
         f"- **Suggested conservative block length: {suggested_s//60} min ({suggested_s} s)** "
         f"= longest sustained dependence among primary classes.",
         "", "## Class temporal dependence", "",
         "| class | primary | interval | bins | lag1 | first<0.2 | sustained<0.2 | detrended sustained |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in dep_rows:
        L.append(f"| {r['class']} | {r['is_primary']} | {r['train_start']} .. {r['train_end']} | "
                 f"{r['n_bins']} | {r['lag1_acf']:.3f} | {r['first_below_0.2_raw']} | "
                 f"{r['sustained_below_0.2_raw']} | {r['sustained_below_0.2_detrended']} |")
    L += ["", "## Sensitivity (first / sustained lag where |r|<0.2)", "",
          "| class | raw | detrended | log1p |", "| --- | --- | --- | --- |"]
    for r in sens_rows:
        L.append(f"| {r['class']} | {r['raw_first']}/{r['raw_sustained']} | "
                 f"{r['detrended_first']}/{r['detrended_sustained']} | {r['log1p_first']}/{r['log1p_sustained']} |")
    L += ["", "## Blocks available (per class)", "",
          "| class | " + " | ".join(f"{b//60}min" for b in BLOCK_CANDIDATES_S) + " |",
          "| --- | " + " | ".join("---" for _ in BLOCK_CANDIDATES_S) + " |"]
    for r in dep_rows:
        L.append(f"| {r['class']} | " + " | ".join(str(r[f"blocks_at_{b}s"]) for b in BLOCK_CANDIDATES_S) + " |")
    L += ["", "## Notes", "",
          "- Bootstrap (later) must resample WITHIN each class's own temporal sequence; classes do",
          "  not share contemporaneous blocks.",
          "- Supplementary classes (mitm, ransomware) reported separately; their intervals are short",
          "  (~2 h), so few blocks are available at long block lengths."]
    (out / "temporal_autocorrelation_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[acf] suggested block {suggested_s}s; wrote {out}")


if __name__ == "__main__":
    main()
