"""Bounded Step-1.2 diagnostics.

Subcommands
-----------
cosine-ablation : compare a baseline vs ablated standardised mean-profile cosine.
l7-proto        : characterise fractional L7_PROTO observations.

These are read-only analyses: they never modify the frozen baseline in
``outputs/unsw_full`` and they do not alter the similarity definitions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

L7_COLUMNS = ["L7_PROTO", "Attack", "Label", "PROTOCOL", "TCP_FLAGS",
              "IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]


def _pairs(mat: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append((labels[i], labels[j], float(mat[i, j])))
    return pd.DataFrame(rows, columns=["class_a", "class_b", "value"])


def _ranks(df: pd.DataFrame) -> pd.Series:
    return df["value"].rank(ascending=False, method="min")


def cosine_ablation(baseline: Path, ablation: Path, out: Path) -> None:
    base = pd.read_csv(baseline, index_col=0)
    abla = pd.read_csv(ablation, index_col=0)
    labels = list(base.index)
    if list(abla.index) != labels:
        raise SystemExit("baseline and ablation class order differ")

    b = _pairs(base.to_numpy(), labels).rename(columns={"value": "cosine_baseline"})
    a = _pairs(abla.to_numpy(), labels).rename(columns={"value": "cosine_ablation"})
    cmp = b.merge(a, on=["class_a", "class_b"])
    cmp["delta"] = cmp["cosine_ablation"] - cmp["cosine_baseline"]

    top5_b = set(map(tuple, cmp.sort_values("cosine_baseline", ascending=False)
                     .head(5)[["class_a", "class_b"]].to_numpy()))
    top5_a = set(map(tuple, cmp.sort_values("cosine_ablation", ascending=False)
                     .head(5)[["class_a", "class_b"]].to_numpy()))
    rho = spearmanr(cmp["cosine_baseline"], cmp["cosine_ablation"]).statistic
    tau = kendalltau(cmp["cosine_baseline"], cmp["cosine_ablation"]).statistic

    def generic_row(df, col):
        m = df[(df["class_a"] == "Generic") | (df["class_b"] == "Generic")]
        return float(m[col].mean())

    out.mkdir(parents=True, exist_ok=True)
    cmp.sort_values("cosine_ablation", ascending=False).to_csv(
        out / "cosine_ablation_deltas.csv", index=False)

    lines = [
        "# Cosine ablation: 37 vs 35 magnitude features",
        "",
        "Baseline excludes nothing; ablation drops `SRC_TO_DST_SECOND_BYTES` and "
        "`DST_TO_SRC_SECOND_BYTES`. Cosine is otherwise identical.",
        "",
        f"- Pairs compared: {len(cmp)}",
        f"- Mean |delta|: {cmp['delta'].abs().mean():.4f}",
        f"- Max |delta|: {cmp['delta'].abs().max():.4f} "
        f"({cmp.loc[cmp['delta'].abs().idxmax(), 'class_a']}-"
        f"{cmp.loc[cmp['delta'].abs().idxmax(), 'class_b']})",
        f"- Spearman rho: {rho:.4f}",
        f"- Kendall tau: {tau:.4f}",
        f"- Top-5 pair overlap: {len(top5_b & top5_a)}/5",
        f"- Generic mean similarity: baseline {generic_row(cmp, 'cosine_baseline'):.4f} "
        f"-> ablated {generic_row(cmp, 'cosine_ablation'):.4f}",
        "",
        "| rank | baseline pair | cos | ablated pair | cos |",
        "| --- | --- | --- | --- | --- |",
    ]
    ob = cmp.sort_values("cosine_baseline", ascending=False).head(5).reset_index(drop=True)
    oa = cmp.sort_values("cosine_ablation", ascending=False).head(5).reset_index(drop=True)
    for k in range(5):
        lines.append(f"| {k + 1} | {ob.loc[k, 'class_a']}-{ob.loc[k, 'class_b']} | "
                     f"{ob.loc[k, 'cosine_baseline']:.4f} | "
                     f"{oa.loc[k, 'class_a']}-{oa.loc[k, 'class_b']} | "
                     f"{oa.loc[k, 'cosine_ablation']:.4f} |")
    lines += ["", "## Largest changes", "",
              "| pair | baseline | ablated | delta |", "| --- | --- | --- | --- |"]
    for r in cmp.reindex(cmp["delta"].abs().sort_values(ascending=False).index).head(8).itertuples():
        lines.append(f"| {r.class_a}-{r.class_b} | {r.cosine_baseline:.4f} | "
                     f"{r.cosine_ablation:.4f} | {r.delta:+.4f} |")
    (out / "cosine_ablation_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def l7_proto(data: Path, out: Path, chunk: int = 250_000) -> None:
    frac_counts: Counter = Counter()
    by_class: dict[str, list[int]] = {}   # class -> [total, fractional, nan]
    co = Counter()                        # co-occurrence among fractional rows
    base = Counter()                      # same conditions among all rows
    cond_cols = ["TCP_FLAGS", "IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]
    frac_min, frac_max = np.inf, -np.inf

    reader = pd.read_csv(data, usecols=L7_COLUMNS, chunksize=chunk)
    for ch in reader:
        v = ch["L7_PROTO"].to_numpy(dtype="float64")
        finite = np.isfinite(v)
        frac = finite & (v != np.floor(v))
        cls = ch["Attack"].to_numpy()
        for c in np.unique(cls):
            m = cls == c
            rec = by_class.setdefault(str(c), [0, 0, 0])
            rec[0] += int(m.sum())
            rec[1] += int((m & frac).sum())
            rec[2] += int((m & ~finite).sum())
        fv = v[frac]
        if fv.size:
            frac_min = min(frac_min, float(fv.min()))
            frac_max = max(frac_max, float(fv.max()))
            vals, cnts = np.unique(np.round(fv, 6), return_counts=True)
            for val, cnt in zip(vals, cnts):
                frac_counts[float(val)] += int(cnt)
        for col in cond_cols:
            col_v = pd.to_numeric(ch[col], errors="coerce").to_numpy(dtype="float64")
            zero = np.isfinite(col_v) & (col_v == 0.0)
            co[col] += int((zero & frac).sum())
            base[col] += int(zero.sum())
    total_rows = sum(r[0] for r in by_class.values())
    total_frac = sum(r[1] for r in by_class.values())

    out.mkdir(parents=True, exist_ok=True)
    vals = pd.DataFrame(sorted(frac_counts.items()), columns=["l7_value", "count"])
    vals.to_csv(out / "l7_proto_fractional_values.csv", index=False)
    rows = [{"class": c, "rows": r[0], "l7_nan": r[2], "fractional": r[1],
             "fractional_pct": 100.0 * r[1] / r[0] if r[0] else 0.0}
            for c, r in by_class.items()]
    by = pd.DataFrame(rows).sort_values("fractional", ascending=False)
    by.to_csv(out / "l7_proto_fractional_by_class.csv", index=False)

    lines = [
        "# Fractional `L7_PROTO` investigation (NF-UNSW-NB15-v3)",
        "",
        f"- Rows scanned: {total_rows:,}",
        f"- Fractional (non-integral, finite) values: {total_frac:,} "
        f"({100.0 * total_frac / total_rows:.4f}%)",
        f"- Distinct fractional values: {len(frac_counts)} "
        f"(range {frac_min:.4f}..{frac_max:.4f})",
        f"- NaN L7_PROTO: {sum(r[2] for r in by_class.values()):,}",
        "",
        "## Fractional values", "", "| value | count |", "| --- | --- |",
    ]
    for r in vals.sort_values("count", ascending=False).head(20).itertuples():
        lines.append(f"| {r.l7_value} | {r.count:,} |")
    lines += ["", "## By class", "",
              "| class | rows | fractional | % |", "| --- | --- | --- | --- |"]
    for r in by.to_dict("records"):
        lines.append(f"| {r['class']} | {r['rows']:,} | {r['fractional']:,} | {r['fractional_pct']:.3f} |")
    lines += ["", "## Co-occurrence among fractional rows", "",
              "| condition | fractional rows | share | all-row base rate |", "| --- | --- | --- | --- |"]
    for col in cond_cols:
        share = co[col] / total_frac if total_frac else 0.0
        brate = base[col] / total_rows if total_rows else 0.0
        lines.append(f"| {col} == 0 | {co[col]:,} | {share:.3f} | {brate:.3f} |")
    (out / "l7_proto_fractional.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ca = sub.add_parser("cosine-ablation")
    ca.add_argument("--baseline", required=True, type=Path)
    ca.add_argument("--ablation", required=True, type=Path)
    ca.add_argument("--out", required=True, type=Path)
    l7 = sub.add_parser("l7-proto")
    l7.add_argument("--data", required=True, type=Path)
    l7.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "cosine-ablation":
        cosine_ablation(args.baseline, args.ablation, args.out)
    else:
        l7_proto(args.data, args.out)


if __name__ == "__main__":
    main()
