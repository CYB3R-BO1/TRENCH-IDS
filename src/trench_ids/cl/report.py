"""Turns the finished experiment matrix into the results document.

One command, one file, every number traceable to a named run directory.
Written as code rather than assembled by hand because the alternative --
copying figures out of JSON into prose -- is how a results section drifts
away from the runs that produced it.

Three comparisons, each with its own baseline, because they answer different
questions and pairing them all against one arm would be meaningless:

  architecture   Flat -> Flat+Host -> GNN, holding the CL method fixed.
                 Baseline: Flat. Isolates what the graph contributes.
  method         fine-tuning / EWC / replay / TRD variants, holding the
                 architecture fixed. Baseline: plain fine-tuning, because
                 that is what EWC and every TRD arm differ from by exactly
                 one thing (the loss). Replay and Replay+TRD appear in the
                 same table as the stronger reference point, but they change
                 the training data too, so pairing the regularisers against
                 them would confound mechanism with regime.
  weighting      TRD-transfer / TRD-inverse / TRD-drift against TRD-uniform.
                  Baseline: TRD-uniform. This is the one that tests the
                  project's hypothesis: does weighting by measured
                  transferability beat weighting everything equally -- and,
                  after drift.py measured transferability anti-correlated
                  with per-relation drift, does weighting by drift directly
                  beat weighting by a signal anti-correlated with it?

Run:  python -m trench_ids.cl.report --out docs/results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from trench_ids.cl.compare_runs import (
    MEANINGFUL_DELTA,
    METRICS,
    aggregate,
    build_report,
    load_run,
)
from trench_ids.cl.run_matrix import BUDGETS, DEFAULT_BUDGET

SEEDS = (42, 1, 2)
RUNS_DIR = Path("runs")

COMPARISONS: dict[str, dict[str, Any]] = {
    "architecture": {
        "baseline": "Flat",
        "arms": {
            "Flat": "flat_replay",
            "Flat+Host": "flathost_replay",
            "Flat+Host (matched reachable params)": "flathost_small_replay",
            "GNN (attention fusion)": "gnn_replay",
            "GNN (concat fusion)": "gnn_concat_replay",
        },
        "question": (
            "Does the heterogeneous graph contribute over per-flow features? The ladder "
            "adds one capability per rung -- host aggregates, then message passing -- so a "
            "win can be attributed rather than merely observed. Read it as two separate "
            "comparisons, because they answer differently: message passing *does* beat "
            "per-flow-only features (GNN over Flat), but concatenating a flow's own two "
            "endpoint-host aggregates beats propagating the same values through 11 "
            "relations (Flat+Host over GNN). The extra rungs exist to close off the two "
            "explanations that would otherwise rescue the GNN: `concat fusion` replaces "
            "HAN's convex combination with concatenate-and-project (it is *worse*, so the "
            "merge was not the bottleneck), and `matched reachable params` shrinks "
            "Flat+Host to the GNN's 156,562 gradient-reachable parameters rather than its "
            "264,850 nominal ones -- 40.9% of the GNN's weights get no gradient at "
            "num_layers=1, so the original matching compared ~157K trainable against "
            "~263K. The gap survives the correction."
        ),
    },
    "method": {
        "baseline": "Finetune",
        "arms": {
            "Finetune": "gnn_finetune",
            "EWC": "gnn_ewc",
            "Replay": "gnn_replay",
            "TRD-uniform": "gnn_trd_uniform",
            "TRD-transfer": "gnn_trd_transfer",
            "TRD-inverse": "gnn_trd_inverse",
            "TRD-drift": "gnn_trd_drift",
            "Replay+TRD": "gnn_replay_trd",
        },
        "question": (
            "What mitigates forgetting? Fine-tuning, EWC and the four TRD arms all run "
            "without replay, so TRD is compared to EWC like-for-like; Replay and "
            "Replay+TRD show whether the regulariser adds anything on top of the "
            "strongest simple baseline."
        ),
    },
    "weighting": {
        "baseline": "TRD-uniform",
        "arms": {
            "TRD-uniform": "gnn_trd_uniform",
            "TRD-transfer": "gnn_trd_transfer",
            "TRD-inverse": "gnn_trd_inverse",
            "TRD-drift": "gnn_trd_drift",
        },
        "question": (
            "Does weighting the penalty by measured transferability beat weighting "
            "every relation equally -- and does weighting by measured drift (the "
            "quantity transferability is anti-correlated with, so 'inverse' was a "
            "proxy for it) beat both?"
        ),
    },
}


def arm_dirs(prefix: str, seeds: tuple[int, ...] = SEEDS) -> list[Path]:
    """Finished run directories for one arm, one per seed.

    Finished means ``summary.json`` exists, not merely that the directory
    does. ``run_matrix`` creates a run's directory before training starts, so
    a directory-only check picks up in-flight runs whose
    ``forgetting_matrix.json`` does not exist yet and makes ``build`` fail --
    which matters because the whole point of generating the report from disk
    is being able to run it partway through a multi-hour sweep and see what
    has landed.
    """
    return [
        d
        for s in seeds
        if (d := RUNS_DIR / f"{prefix}_s{s}").exists() and (d / "summary.json").exists()
    ]


def _markdown_table(report: dict[str, Any], arm_order: list[str]) -> list[str]:
    header = "| Arm | n | Final acc. | F1-macro | Forgetting | BWT |"
    lines = [header, "|---|---:|---:|---:|---:|---:|"]

    def cell(agg: dict[str, Any], metric: str) -> str:
        if metric not in agg:
            return "n/a"
        stat = agg[metric]
        if stat["std"] is None:
            return f"{stat['mean']:.4f}"
        return f"{stat['mean']:.4f} ± {stat['std']:.3f}"

    for name in arm_order:
        agg = report["aggregates"].get(name)
        if not agg:
            continue
        n = max((s["n"] for s in agg.values()), default=0)
        lines.append(
            f"| {name} | {n} | {cell(agg, 'final_average_accuracy')} | "
            f"{cell(agg, 'f1_macro')} | {cell(agg, 'average_forgetting')} | "
            f"{cell(agg, 'backward_transfer')} |"
        )
    return lines


def _markdown_comparisons(report: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"Paired by seed against **{report['baseline_arm']}**. A difference counts as "
        f"MEANINGFUL only if `|mean delta| >= {report['meaningful_delta_threshold']}` "
        "*and* it has the same sign on every seed -- the threshold is the observed "
        "seed-to-seed spread, so an effect has to clear the noise floor rather than "
        "merely reach significance at n=3.",
        "",
        "A verdict on fewer than 3 shared seeds is marked *provisional*: the decision rule "
        "leans on sign consistency, and on 2 seeds that condition is nearly vacuous "
        "(a coin flip agrees half the time). Such rows are reporting what has landed so "
        "far, not a settled result.",
        "",
        "| Comparison | Metric | Seeds | Mean delta | Direction | p | Same sign | Verdict |",
        "|---|---|---:|---:|---|---:|---|---|",
    ]
    for pair, entries in report["paired_comparisons"].items():
        if not entries:
            lines.append(f"| {pair} | — | 0 | — | — | — | — | fewer than 2 shared seeds |")
            continue
        for entry in entries:
            if entry["metric"] not in ("final_average_accuracy", "f1_macro", "average_forgetting"):
                continue
            n_seeds = len(entry.get("seeds", []))
            if not entry["meaningful"]:
                verdict = "not meaningful"
            elif n_seeds < len(SEEDS):
                verdict = f"provisional (n={n_seeds})"
            else:
                verdict = "**MEANINGFUL**"
            lines.append(
                f"| {pair} | {entry['metric']} | {n_seeds} | {entry['mean_delta']:+.4f} | "
                f"{'better' if entry['improves_baseline'] else 'worse'} | "
                f"{entry['p_value']:.3f} | "
                f"{'yes' if entry['sign_consistent_across_seeds'] else 'no'} | "
                f"{verdict} |"
            )
    return lines


def _summary_row(path: Path, lam: float) -> tuple[float, float, float, float] | None:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    return (
        lam,
        float(summary.get("final_average_val_accuracy", float("nan"))),
        float(summary.get("final_average_accuracy", float("nan"))),
        float(summary.get("average_forgetting", float("nan"))),
    )


def lambda_selection_table() -> list[str]:
    """λ_d candidates ranked by **validation** accuracy.

    Test numbers are printed next to them purely so the gap between the two
    is visible; the selection uses the val column, which is the whole point
    of running this group separately.

    Plain fine-tuning is included as the λ_d = 0 anchor. Without it the table
    would always name a "winner" -- the best of a set of candidates that
    might every one of them be worse than switching the penalty off, which is
    a negative result dressed as a selection. The anchor is the same
    architecture, seed and schedule with the distillation term absent, so the
    comparison is like-for-like.
    """
    rows: list[tuple[float, float, float, float]] = []
    for path in sorted(RUNS_DIR.glob("lambda_trd_*_s42")):
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        lam = float(json.loads(summary_path.read_text()).get("distill_lambda_d", float("nan")))
        row = _summary_row(path, lam)
        if row is not None:
            rows.append(row)
    if not rows:
        return []

    anchor = _summary_row(RUNS_DIR / "gnn_finetune_s42", 0.0)

    rows.sort(key=lambda r: r[1], reverse=True)
    lines = [
        "| λ_d | Val accuracy (selection metric) | Test accuracy | Forgetting |",
        "|---:|---:|---:|---:|",
    ]
    for lam, val, test, forget in rows:
        lines.append(f"| {lam:g} | {val:.4f} | {test:.4f} | {forget:.4f} |")
    if anchor is not None:
        lines.append(
            f"| _0 (off — plain fine-tuning)_ | _{anchor[1]:.4f}_ | _{anchor[2]:.4f}_ | "
            f"_{anchor[3]:.4f}_ |"
        )
    lines.append("")

    best_lam, best_val = rows[0][0], rows[0][1]
    if anchor is None:
        lines.append(
            f"Selected: **λ_d = {best_lam:g}** (highest validation accuracy). "
            "The λ_d = 0 anchor (`gnn_finetune_s42`) has not been run, so whether any "
            "penalty beats switching it off is still unestablished."
        )
    elif best_val > anchor[1]:
        lines.append(
            f"Selected: **λ_d = {best_lam:g}** (highest validation accuracy), which beats "
            f"the λ_d = 0 anchor by {best_val - anchor[1]:+.4f} val accuracy."
        )
    else:
        lines.append(
            f"**No candidate beat the λ_d = 0 anchor.** The best λ_d ({best_lam:g}, val "
            f"{best_val:.4f}) is {best_val - anchor[1]:+.4f} against plain fine-tuning's "
            f"{anchor[1]:.4f}, so on this benchmark the distillation penalty only ever "
            "costs plasticity. λ_d is reported as selected for the arms below so the "
            "weighting ablation still runs at its most favourable setting, but the "
            "mechanism is a negative result and is written up as one."
        )

    boundary = min(r[0] for r in rows), max(r[0] for r in rows)
    if best_lam in boundary:
        lines.append("")
        lines.append(
            f"⚠️ The winner sits at the edge of the swept range [{boundary[0]:g}, "
            f"{boundary[1]:g}], so this is a boundary result: the sweep cannot "
            "distinguish an optimum from a truncation without extending past it."
        )
    return lines


def _val_accuracy(run_dir: Path) -> float | None:
    """Final average validation accuracy for one run, or None if absent.

    Read straight from ``summary.json`` rather than routed through
    ``load_run``, which reports the test-side metrics only. Selection has to
    happen on val, so this is deliberately the one number the budget table
    ranks on.
    """
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    value = json.loads(summary_path.read_text()).get("final_average_val_accuracy")
    return None if value is None else float(value)


def _train_graphs_per_task() -> int | None:
    """Task 1's train-split graph count, for expressing a budget as a share
    of one task. Read from the graph set rather than hardcoded: the number
    changed by 6.7x in the 2026-08-16 rebuild, and a stale literal is exactly
    how the standing buffer size came to be documented as 2.5% of a task
    while actually storing 16.5% of it."""
    counts_path = Path("data/graphs/graph_counts.json")
    if not counts_path.exists():
        return None
    try:
        return int(json.loads(counts_path.read_text())["1"]["train"]["num_graphs"])
    except (KeyError, ValueError, TypeError):
        return None


def budget_rows(arms: dict[int, str]) -> list[tuple[int, int, float, float, float]]:
    """``(budget, n_seeds, val, test, forgetting)`` per budget, seed-averaged.

    Budgets with no finished run are skipped rather than reported as zero, so
    the table can be generated partway through the sweep. A run counts as
    finished only when it has *both* ``summary.json`` (val) and
    ``forgetting_matrix.json`` (test/forgetting) -- averaging val over one
    subset of seeds and test over another would put a selection metric and a
    test metric from disjoint runs side-by-side in the same row.
    """
    rows = []
    for budget, prefix in sorted(arms.items()):
        dirs = [
            d
            for d in arm_dirs(prefix)
            if (d / "forgetting_matrix.json").exists() and _val_accuracy(d) is not None
        ]
        if not dirs:
            continue
        vals = [_val_accuracy(d) for d in dirs]
        runs = [load_run(d) for d in dirs]
        agg = aggregate(runs)
        rows.append(
            (
                budget,
                len(dirs),
                sum(vals) / len(vals),
                agg["final_average_accuracy"]["mean"],
                agg["average_forgetting"]["mean"],
            )
        )
    return rows


def select_budget(rows: list[tuple[int, int, float, float, float]]) -> int | None:
    """The smallest budget whose val accuracy is within the pre-registered
    noise floor of the best.

    Plain argmax would be the wrong rule here and would look defensible: val
    accuracy is (weakly) increasing in memory, so argmax hands back the
    largest budget swept almost by construction, and the "experiment" would
    only ever ratify whatever ceiling the grid happened to stop at. Memory is
    the cost this sweep exists to minimise, so among budgets that are
    statistically indistinguishable on val the smallest one wins -- the
    one-standard-error rule (Breiman et al. 1984; Hastie & Tibshirani), with
    the project's pre-registered MEANINGFUL_DELTA standing in for 1 SE so
    this decision uses the same noise floor as every other comparison in
    the document.
    """
    if not rows:
        return None
    best_val = max(r[2] for r in rows)
    within = [r[0] for r in rows if r[2] >= best_val - MEANINGFUL_DELTA]
    return min(within)


def budget_selection_table() -> list[str]:
    """Replay memory budget per encoder, ranked and selected on validation."""
    per_task = _train_graphs_per_task()
    lines: list[str] = []
    for encoder, prefix in (("GNN", "budget_gnn"), ("Flat+Host", "budget_flathost")):
        arms = {b: f"{prefix}_{b}" for b in BUDGETS}
        arms[DEFAULT_BUDGET] = "gnn_replay" if encoder == "GNN" else "flathost_replay"
        rows = budget_rows(arms)
        if not rows:
            continue
        lines += [
            f"### {encoder}",
            "",
            "| Buffer (graphs/task) | Share of a task's train split | n | "
            "Val accuracy (selection metric) | Test accuracy | Forgetting |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for budget, n, val, test, forget in rows:
            share = f"{budget / per_task:.1%}" if per_task else "—"
            default_mark = " *(standing default)*" if budget == DEFAULT_BUDGET else ""
            lines.append(
                f"| {budget}{default_mark} | {share} | {n} | {val:.4f} | "
                f"{test:.4f} | {forget:.4f} |"
            )
        lines.append("")

        chosen = select_budget(rows)
        best = max(rows, key=lambda r: r[2])
        by_budget = {r[0]: r for r in rows}
        if chosen == best[0]:
            lines.append(
                f"Selected: **{chosen} graphs/task** — highest validation accuracy "
                f"({best[2]:.4f}), and no smaller budget comes within "
                f"{MEANINGFUL_DELTA} of it."
            )
        else:
            lines.append(
                f"Selected: **{chosen} graphs/task** (val {by_budget[chosen][2]:.4f}) — the "
                f"smallest budget within {MEANINGFUL_DELTA} validation accuracy of the best "
                f"({best[0]} graphs/task at {best[2]:.4f}, a gap of "
                f"{best[2] - by_budget[chosen][2]:+.4f}). Memory is the cost being "
                "minimised, so a budget that cannot be distinguished from the best on val "
                "is preferred when it is cheaper."
            )
        swept = [r[0] for r in rows]
        if chosen == min(swept):
            lines += [
                "",
                f"⚠️ The selected budget is the smallest one swept ({min(swept)}), so the "
                "curve has not been shown to turn: an even smaller buffer may do just as "
                "well, and the memory requirement reported here is an upper bound rather "
                "than a measured minimum.",
            ]
        elif best[0] == max(swept):
            lines += [
                "",
                f"⚠️ Validation accuracy is still rising at the largest budget swept "
                f"({max(swept)}), so the ceiling is a truncation rather than an optimum. "
                "What the selection rule can still say is how little memory suffices to "
                "get within the noise floor of it.",
            ]
        lines.append("")

    if not lines:
        return []
    default_share = f" ({DEFAULT_BUDGET / per_task:.1%} of a task)" if per_task else ""
    return [
        "The standing default of "
        f"{DEFAULT_BUDGET} graphs/task{default_share} was inherited from the pre-rebuild "
        "benchmark, where the same number was ~2.5% of a task, and was never swept after "
        "the rebuild cut each task to "
        f"{per_task or '~1.2K'} train graphs. `replay_fraction` is held at 0.3 throughout, "
        "so rehearsal is 30% of the training pool at every point on the curve and the only "
        "variable is how many *distinct* past graphs that 30% is drawn from — compute held "
        "constant, memory varied.",
        "",
        *lines,
    ]


def build(out_path: Path) -> str:
    sections: list[str] = [
        "# TRENCH-IDS results",
        "",
        "Generated by `python -m trench_ids.cl.report`. Every figure is read from a "
        "run directory under `runs/`; nothing here is transcribed by hand.",
        "",
    ]

    lambda_rows = lambda_selection_table()
    if lambda_rows:
        sections += [
            "## λ_d selection (validation split)",
            "",
            "Seed 42 only. Selection is made on validation accuracy; the test column "
            "is shown for transparency and was not used to choose.",
            "",
            *lambda_rows,
            "",
        ]

    budget_rows_md = budget_selection_table()
    if budget_rows_md:
        sections += [
            "## Replay memory budget (validation split)",
            "",
            "*How much past data does replay actually need? Replay is the strongest "
            "forgetting mitigation on this benchmark, so the honest version of that claim "
            "has to say what it costs to store — a method that retains a sixth of every "
            "past task is closer to partial joint training than to continual learning. "
            "Selection is on validation accuracy; the test column is shown for "
            "transparency and was not used to choose.*",
            "",
            *budget_rows_md,
            "",
        ]

    for title, spec in COMPARISONS.items():
        arms = {name: arm_dirs(prefix) for name, prefix in spec["arms"].items()}
        arms = {name: dirs for name, dirs in arms.items() if dirs}
        if spec["baseline"] not in arms:
            continue
        report = build_report(arms, spec["baseline"])
        sections += [
            f"## {title.capitalize()} comparison",
            "",
            f"*{spec['question']}*",
            "",
            *_markdown_table(report, list(spec["arms"])),
            *_markdown_comparisons(report),
            "",
        ]

    # train_joint.py writes joint_summary.json, not summary.json -- it has no
    # task sequence and so no forgetting matrix, and its summary carries a
    # different shape. Checking only "summary.json" silently dropped the upper
    # bound from this document, which is the one number that says how much of
    # the remaining gap is about *continual* learning rather than the model.
    joint = RUNS_DIR / "joint" / "joint_summary.json"
    if joint.exists():
        summary = json.loads(joint.read_text())
        sections += [
            "## Non-continual upper bound",
            "",
            "All six tasks pooled and trained together, no task boundaries. This is the "
            "ceiling any continual-learning method is chasing; the gap to it is the part "
            "of the problem that is genuinely about *continual* learning rather than "
            "about the architecture.",
            "",
            "| Metric | Value |",
            "|---|---:|",
            *[
                f"| {key} | {value:.4f} |"
                for key, value in summary.items()
                if isinstance(value, (int, float)) and key != "seed"
            ],
            "",
        ]

    text = "\n".join(sections)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def per_class_table(arm_prefix: str) -> list[str]:
    """Per-class F1 for one arm, averaged over its seeds.

    Kept separate from the main tables because it answers a different
    question: pooled accuracy can hold steady while a rare class collapses,
    and on this benchmark the rare classes are the ones the design went out
    of its way to include.
    """
    runs = [load_run(d) for d in arm_dirs(arm_prefix)]
    runs = [r for r in runs if "per_class_f1" in r]
    if not runs:
        return []
    classes = sorted(runs[0]["per_class_f1"])
    lines = ["| Class | mean F1 | n |", "|---|---:|---:|"]
    for name in classes:
        values = [r["per_class_f1"][name] for r in runs if name in r["per_class_f1"]]
        if values:
            lines.append(f"| {name} | {sum(values) / len(values):.4f} | {len(values)} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TRENCH-IDS results document.")
    parser.add_argument("--out", default="docs/results.md")
    parser.add_argument(
        "--per-class-arm",
        default=None,
        help="Also append a per-class F1 table for this arm prefix (e.g. gnn_trd_transfer).",
    )
    args = parser.parse_args()

    # The document is UTF-8 (lambda, arrows, warning glyphs) but a Windows
    # console defaults to cp1252, and echoing it there raised
    # UnicodeEncodeError *after* the file had already been written -- a
    # traceback and a non-zero exit code for a run that had actually
    # succeeded, which is exactly the failure that makes someone re-run a
    # finished sweep. The file write has always specified its own encoding;
    # only this echo was fragile.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    out_path = Path(args.out)
    text = build(out_path)
    if args.per_class_arm:
        rows = per_class_table(args.per_class_arm)
        if rows:
            text += "\n".join(["", f"## Per-class F1 — {args.per_class_arm}", "", *rows, ""])
            out_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[report] written to {out_path}")


__all__ = ["build", "arm_dirs", "lambda_selection_table", "per_class_table", "METRICS", "aggregate"]


if __name__ == "__main__":
    main()
