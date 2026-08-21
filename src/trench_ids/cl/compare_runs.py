"""Aggregate the three-arm ablation into one comparison table + paired tests.

Consumes finished run directories and produces the table the ablation is
reported from. Every metric is recomputed from on-disk artifacts
(``forgetting_matrix.json`` and ``eval/pooled_final_metrics.json``) rather
than read out of ``summary.json``, so runs produced before a given metric
existed -- e.g. the pre-existing GNN replay runs, which predate
``backward_transfer`` -- are handled identically to new ones with no
special-casing and no re-training.

The statistical treatment is **paired by seed**, not unpaired. Every arm is
run on the same seed set, and seed is by far the largest nuisance factor
(the existing replay runs vary by ~0.015 accuracy across seeds, which is
larger than most between-arm differences worth caring about). Pairing
removes that shared variance; an unpaired t-test on n=3 would mostly be
measuring it.

Read ``docs/pre-registration-2026-08-07-flat-vs-gnn.md`` before interpreting
anything this prints -- the decision rule, the meaningful-effect threshold,
and the power limitation were all fixed before any run finished.

Run: python -m trench_ids.cl.compare_runs \\
       --arm "Flat=runs/flat_replay_seed42,runs/flat_replay_seed1,runs/flat_replay_seed2" \\
       --arm "Flat+Host=runs/flathost_replay_seed42,..." \\
       --arm "GNN=runs/gnn_replay_seed42,..." \\
       --baseline Flat --out-dir runs/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scipy import stats

from trench_ids.cl.train import average_forgetting, backward_transfer, final_average_accuracy

# Reported for every arm. "higher_is_better" drives the sign of the
# improvement column so a reader never has to remember which way forgetting
# points.
METRICS: dict[str, bool] = {
    "final_average_accuracy": True,
    "pooled_accuracy": True,
    "f1_macro": True,
    "f1_weighted": True,
    "average_forgetting": False,
    "backward_transfer": True,
}

# Pre-registered: a paired mean difference must exceed this to be called
# meaningful, regardless of p-value. Set from the observed seed-to-seed
# spread of the existing replay runs (~0.015), so an effect has to clear
# the noise floor rather than merely reach significance on n=3.
MEANINGFUL_DELTA = 0.02


def load_run(run_dir: Path) -> dict[str, Any]:
    """Every reported metric for one run, recomputed from its artifacts."""
    matrix_path = run_dir / "forgetting_matrix.json"
    if not matrix_path.exists():
        raise FileNotFoundError(f"{matrix_path} missing -- did this run finish?")
    raw = json.loads(matrix_path.read_text())
    # JSON object keys are strings; the metric helpers index by int.
    matrix = {int(k): {int(j): v for j, v in row.items()} for k, row in raw.items()}

    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "final_average_accuracy": final_average_accuracy(matrix),
        "average_forgetting": average_forgetting(matrix),
        "backward_transfer": backward_transfer(matrix),
    }

    pooled_path = run_dir / "eval" / "pooled_final_metrics.json"
    if pooled_path.exists():
        pooled = json.loads(pooled_path.read_text())
        record["pooled_accuracy"] = pooled["accuracy"]
        record["f1_macro"] = pooled["f1_macro"]
        record["f1_weighted"] = pooled["f1_weighted"]
        record["per_class_f1"] = {
            name: stat["f1"] for name, stat in pooled["per_class"].items()
        }

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for key in ("seed", "runtime_seconds", "encoder_parameters", "arm"):
            if key in summary:
                record[key] = summary[key]
    return record


def aggregate(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """mean / sample-std / n per metric across an arm's runs. Sample std is
    undefined for a single run, reported as ``None`` rather than 0.0 so a
    one-seed arm cannot be mistaken for a zero-variance one."""
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [r[metric] for r in runs if metric in r]
        if not values:
            continue
        out[metric] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }
    return out


def paired_comparison(
    baseline: list[dict[str, Any]], other: list[dict[str, Any]], metric: str
) -> dict[str, Any] | None:
    """Seed-paired comparison of one metric between two arms.

    Runs are paired by ``seed``; any seed missing from either arm is
    dropped, so a partially-finished sweep degrades to the seeds both arms
    actually share instead of silently comparing different seed sets.
    Returns ``None`` when fewer than two seeds are shared (nothing to test).
    """
    by_seed_base = {r["seed"]: r for r in baseline if "seed" in r and metric in r}
    by_seed_other = {r["seed"]: r for r in other if "seed" in r and metric in r}
    shared = sorted(set(by_seed_base) & set(by_seed_other))
    if len(shared) < 2:
        return None

    deltas = [by_seed_other[s][metric] - by_seed_base[s][metric] for s in shared]
    higher_is_better = METRICS[metric]
    signed = [d if higher_is_better else -d for d in deltas]
    mean_delta = mean(deltas)

    result = stats.ttest_rel(
        [by_seed_other[s][metric] for s in shared],
        [by_seed_base[s][metric] for s in shared],
    )
    consistent = all(d > 0 for d in signed) or all(d < 0 for d in signed)
    return {
        "metric": metric,
        "seeds": shared,
        "per_seed_delta": dict(zip(shared, deltas, strict=True)),
        "mean_delta": mean_delta,
        "improves_baseline": mean(signed) > 0,
        "p_value": float(result.pvalue),
        "sign_consistent_across_seeds": consistent,
        # The pre-registered rule: magnitude AND consistency, not p alone.
        # n=3 cannot support a p-value-driven claim on its own.
        "meaningful": abs(mean_delta) >= MEANINGFUL_DELTA and consistent,
    }


def build_report(
    arms: dict[str, list[Path]], baseline_arm: str
) -> dict[str, Any]:
    runs = {name: [load_run(d) for d in dirs] for name, dirs in arms.items()}
    if baseline_arm not in runs:
        raise ValueError(f"baseline arm {baseline_arm!r} not among {sorted(runs)}")

    comparisons: dict[str, list[dict[str, Any]]] = {}
    for name, arm_runs in runs.items():
        if name == baseline_arm:
            continue
        found = [
            c
            for metric in METRICS
            if (c := paired_comparison(runs[baseline_arm], arm_runs, metric)) is not None
        ]
        comparisons[f"{name} vs {baseline_arm}"] = found

    return {
        "baseline_arm": baseline_arm,
        "meaningful_delta_threshold": MEANINGFUL_DELTA,
        "aggregates": {name: aggregate(arm_runs) for name, arm_runs in runs.items()},
        "per_run": runs,
        "paired_comparisons": comparisons,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _cell(agg: dict[str, dict[str, float]], metric: str) -> str:
    """One 'mean ± std' table cell, or 'n/a' for a metric this arm lacks."""
    if metric not in agg:
        return f"{'n/a':>16}"
    stat = agg[metric]
    return f"{_fmt(stat['mean'])} ± {_fmt(stat['std'], 3):>5}".rjust(16)


def format_table(report: dict[str, Any]) -> str:
    header = (
        f"{'Arm':<12} {'n':>2}  {'FinalAcc':>16} {'F1-macro':>16} "
        f"{'Forgetting':>16} {'BWT':>16}"
    )
    lines = [header, "-" * len(header)]
    for name, agg in report["aggregates"].items():
        n = max((s["n"] for s in agg.values()), default=0)
        lines.append(
            f"{name:<12} {n:>2}  {_cell(agg, 'final_average_accuracy')} "
            f"{_cell(agg, 'f1_macro')} {_cell(agg, 'average_forgetting')} "
            f"{_cell(agg, 'backward_transfer')}"
        )
    return "\n".join(lines)


def format_comparisons(report: dict[str, Any]) -> str:
    lines = [
        "",
        f"Paired by seed vs. baseline '{report['baseline_arm']}' "
        f"(meaningful = |mean delta| >= {report['meaningful_delta_threshold']} "
        f"AND same sign on every seed):",
    ]
    for pair, entries in report["paired_comparisons"].items():
        lines.append(f"\n  {pair}")
        if not entries:
            lines.append("    (fewer than 2 shared seeds -- nothing to test yet)")
            continue
        for entry in entries:
            verdict = "MEANINGFUL" if entry["meaningful"] else "not meaningful"
            direction = "better" if entry["improves_baseline"] else "worse"
            lines.append(
                f"    {entry['metric']:<24} delta={entry['mean_delta']:+.4f} "
                f"({direction})  p={entry['p_value']:.3f}  "
                f"consistent={entry['sign_consistent_across_seeds']}  -> {verdict}"
            )
    return "\n".join(lines)


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = ["arm", "run_dir", "seed", *METRICS, "runtime_seconds", "encoder_parameters"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for arm, runs in report["per_run"].items():
            for record in runs:
                # record's own "arm" key (copied by load_run from
                # summary.json, when present) must not win here: train_flat.py
                # writes summary["arm"] as a bare "flat"/"flat+host" model-type
                # label, which cannot distinguish two arms that share a model
                # type but differ in configuration (e.g. flathost_replay vs.
                # flathost_small_replay, the parameter-matching control pair)
                # -- unlike this function's own `arm` key, which is the
                # caller-supplied, unambiguous report-arm name.
                writer.writerow({**record, "arm": arm})


def parse_arm(spec: str) -> tuple[str, list[Path]]:
    if "=" not in spec:
        raise ValueError(f"--arm expects 'Name=dir1,dir2', got {spec!r}")
    name, dirs = spec.split("=", 1)
    return name.strip(), [Path(d.strip()) for d in dirs.split(",") if d.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-arm ablation comparison table.")
    parser.add_argument(
        "--arm", action="append", required=True, help="'Name=run_dir1,run_dir2,...' (repeatable)"
    )
    parser.add_argument("--baseline", required=True, help="Arm name every other arm is paired to.")
    parser.add_argument("--out-dir", default="runs/comparison")
    args = parser.parse_args()

    arms = dict(parse_arm(spec) for spec in args.arm)
    report = build_report(arms, args.baseline)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    write_csv(report, out_dir / "comparison.csv")

    table = format_table(report) + "\n" + format_comparisons(report)
    (out_dir / "comparison.txt").write_text(table)
    print(table)
    print(f"\n[compare] wrote {out_dir}")


if __name__ == "__main__":
    main()
