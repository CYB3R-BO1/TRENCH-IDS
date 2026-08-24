"""Runs the full experiment matrix, and nothing else.

Every claim in the write-up traces back to a named run in here, so the matrix
is code rather than a shell transcript: the exact overrides that produced
``runs/<name>/`` are recoverable months later, and a re-run is idempotent
(a run whose ``summary.json`` already exists is skipped, so an interrupted
sweep resumes instead of restarting).

Groups, and what each is for:

``arch``     Flat / Flat+Host / GNN under identical replay settings. The
             ladder adds exactly one capability per rung -- host aggregates,
             then message passing -- so a win can be attributed rather than
             just observed. This is the question the project had never asked:
             every earlier experiment compared GNN variants to other GNN
             variants.
``method``   Fine-tuning / EWC / replay / the four TRD weightings, all on
             the GNN. Establishes what actually mitigates forgetting, and
             whether weighting the distillation by measured transferability
             beats weighting it uniformly (the project's actual hypothesis),
             inversely (the sign control), or by measured drift itself (the
             reformulation the transferability-vs-drift anti-correlation
             motivates).
``lambda``   λ_d selection for TRD, scored on the **validation** split. Every
             selection decision before 2026-08-16 was made on test numbers;
             the val split existed but was never read by any code.
``budget``   How much past data replay actually needs, scored on **val**.
             The standing ``buffer_size_per_task`` was inherited from the
             pre-rebuild benchmark and never swept -- see ``BUDGETS``.
``joint``    Non-continual upper bound: all six tasks pooled, no task
             boundaries.

Run:  python -m trench_ids.cl.run_matrix --group all
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SEEDS = (42, 1, 2)
RUNS_DIR = Path("runs")


@dataclass(frozen=True)
class Run:
    name: str
    module: str
    overrides: tuple[str, ...] = ()
    group: str = ""
    note: str = ""

    def out_dir(self) -> Path:
        return RUNS_DIR / self.name

    def summary_path(self) -> Path:
        """The file whose existence means "this run finished".

        ``train_joint`` writes ``joint_summary.json`` rather than
        ``summary.json`` -- it has no task sequence, so no forgetting matrix
        and a differently-shaped summary. Checking only ``summary.json`` made
        the joint run non-idempotent: it was re-run on every sweep, and
        ``--dry-run`` reported it as outstanding after it had already finished.
        """
        if self.module.endswith("train_joint"):
            return self.out_dir() / "joint_summary.json"
        return self.out_dir() / "summary.json"

    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            self.module,
            f"paths.out_dir={self.out_dir().as_posix()}",
            *self.overrides,
        ]


# EWC off unless a run is specifically about EWC. Zeroing all three lambdas
# is not the same as leaving them at the config default: the default 1.0
# applies a penalty to all 12 parameter groups, which would confound every
# non-EWC arm with a regulariser nobody asked for.
NO_EWC = ("ewc.lambda_r=0", "ewc.lambda_s=0", "ewc.lambda_u=0")

# Both regularised arms (EWC and TRD) spend their warm-up epochs on the
# plain classification loss, because both need a partially-adapted encoder
# before their per-relation signal (Fisher / S_r) means anything. One epoch
# rather than the historical two, so the regulariser is active for 4 of 5
# epochs instead of 3: at two, the encoder drifts unconstrained through 40%
# of training, which handicaps a mechanism whose whole job is to limit
# drift. Applied identically to EWC and TRD so the comparison between them
# is not confounded by the size of that window. Unregularised arms (replay,
# fine-tuning) are unaffected either way -- with the lambdas at zero their
# warm-up and full-loss epochs run the identical loss.
SHORT_WARMUP = ("train.warmup_epochs=1",)

# Softmax temperature turning S_r into w_r, held identical across the lambda
# sweep and the final arms so the selected lambda stays valid. 0.1 was tried
# first and is too sharp: the observed S_r spread (~0.5) drives it to nearly
# one-hot, which reduces "transfer" to "regularise service_of and nothing
# else" -- and service_of is, measurably, the relation that drifts least
# anyway (0.05 cosine drift across the T5->T6 boundary, versus 0.60 for
# terminated_by), so the penalty was concentrated exactly where it was least
# needed and came out effectively inert. 0.3 keeps a clear ordering while
# leaving every relation a non-negligible share.
TEMPERATURE = ("distill.temperature=0.3",)

# Drift scores live on a different scale than transferability, so the drift
# arm gets its own temperature. The first drift runs used the shared T=0.3
# and it degenerated them: measured post-warm-up D_r spans ~0.1-1.25 across
# relations and tasks -- roughly 3x S_r's ~0.5 spread -- so at T=0.3 the
# softmax came out nearly one-hot on `originates` (seed 42 task 2: w=0.748,
# everything else <=0.09), the exact "one relation carries the whole penalty"
# failure documented for transfer@T=0.1 above. Those runs are kept as
# evidence under runs/_superseded_drift_T03/. T=1.0 matches the softmax to
# D_r's actual spread: ordered like `inverse` predicts (the anti-correlation)
# without collapsing onto a single relation.
DRIFT_TEMPERATURE = ("distill.temperature=1.0",)


# Replay memory budgets, in graphs retained per task. The standing default
# is 200, and its config comment ("~2.5% of one task's ~8135 train graphs")
# was accurate for the pre-rebuild benchmark. The 2026-08-16 quota rebuild
# cut every task to 1,214 train graphs and nobody revisited the number, so
# the default actually stores 16.5% of each past task -- by T6, ~300K past
# flows. That is a large enough memory footprint that "is this continual
# learning or just keeping a sixth of the data?" is a fair question, and it
# has never been answered because this knob has never been swept.
#
# 25/50/100/400 brackets the default on both sides deliberately. A sweep
# whose best value sits at an edge cannot tell an optimum from a truncation
# -- the same trap the lambda_d sweep fell into on its first pass and had to
# be re-run to escape. 400 (33% of a task) is included even though nobody
# would ship it, precisely so that "200 is enough" is a measured interior
# result rather than the largest value anyone happened to try.
#
# 200 itself is NOT re-run here: gnn_replay_s* and flathost_replay_s*
# already are that point at identical settings, and re-running it under a
# second name would only add seed noise to a curve it should anchor.
BUDGETS = (25, 50, 100, 400)

# The value every replay result on record was produced at, supplied by the
# gnn_replay / flathost_replay arms rather than re-run under a budget_* name.
DEFAULT_BUDGET = 200

# replay_fraction stays at its config default across every budget, and that
# is what makes the sweep readable. Rehearsal is 30% of the training pool
# either way, so total gradient steps and pool composition are identical at
# every point on the curve; the only thing changing is how many DISTINCT
# past graphs that 30% is drawn from (build_replay_augmented_set upsamples
# with replacement, so a smaller buffer means more repetition, not less
# rehearsal). Compute held constant, memory varied -- which is the question
# being asked. Sweeping both knobs at once would answer neither.
FIXED_REHEARSAL = ("replay.replay_fraction=0.3",)


def _gnn(name: str, group: str, extra: tuple[str, ...], seed: int, note: str = "") -> Run:
    return Run(
        name=f"{name}_s{seed}",
        module="trench_ids.cl.train",
        overrides=(f"train.seed={seed}", *extra),
        group=group,
        note=note,
    )


def _flat(name: str, group: str, extra: tuple[str, ...], seed: int, note: str = "") -> Run:
    return Run(
        name=f"{name}_s{seed}",
        module="trench_ids.cl.train_flat",
        overrides=(f"train.seed={seed}", *extra),
        group=group,
        note=note,
    )


def build_matrix(lambda_d: float, seeds: tuple[int, ...] = SEEDS) -> list[Run]:
    """Every run, in execution order (cheapest-to-diagnose first)."""
    runs: list[Run] = []

    # --- lambda selection: seed 42 only, scored on val ------------------
    # Run WITHOUT replay, because that is the regime the method group tests
    # TRD in (see below). A lambda selected under replay does not transfer:
    # with replay the encoder is already anchored by old data, so the useful
    # penalty scale is much smaller.
    # Spans 500x. The upper half (2-50) was run first and came out monotonically
    # worse -- val accuracy 0.404 at 0.5 against 0.270 at 2.0 -- so the interesting
    # region is *below* the original floor, not above it. The low candidates were
    # added rather than swapping the range out, because a sweep that only covers
    # the winning region invites exactly the "you didn't tune it hard enough"
    # objection that the 2026-07-21 EWC lambda sweep existed to close off.
    #
    # gnn_finetune is the lambda_d -> 0 anchor for this curve: if no candidate
    # beats it, the honest reading is that the penalty only ever costs
    # plasticity, and TRD is a negative result on this benchmark.
    for candidate in (0.1, 0.25, 0.5, 2.0, 10.0, 50.0):
        runs.append(
            _gnn(
                f"lambda_trd_{candidate:g}",
                "lambda",
                (
                    "replay.enabled=false",
                    *NO_EWC,
                    *SHORT_WARMUP,
                    *TEMPERATURE,
                    "distill.enabled=true",
                    "distill.weighting=transfer",
                    f"distill.lambda_d={candidate}",
                ),
                42,
                note="lambda_d selection, no replay, decided on val accuracy only",
            )
        )

    # --- architecture ladder -------------------------------------------
    for seed in seeds:
        runs.append(
            _flat(
                "flat_replay",
                "arch",
                ("replay.enabled=true", "model.use_host_features=false"),
                seed,
            )
        )
        runs.append(
            _flat(
                "flathost_replay",
                "arch",
                ("replay.enabled=true", "model.use_host_features=true"),
                seed,
            )
        )
        # Matched-EFFECTIVE-capacity control. The arms above are matched to the
        # GNN's *nominal* 264,850 parameters, but a measured 40.9% of those
        # (108,288, the 6 non-Flow relations' rel_lins/combine_lins plus the 4
        # unused non-Flow fusion modules) receive zero gradient from the
        # classification loss at num_layers=1 -- computed but never consumed,
        # since train.py only reads output.fused["flow"]. So the ladder as
        # first run compared ~157K trainable parameters against ~263K, a 1.7x
        # gap in the flat model's favour, in a control whose entire purpose
        # was to stop capacity confounding the architecture question.
        #
        # mlp_hidden=123 brings Flat+Host to 156,577 parameters against the
        # GNN's 156,562 reachable ones (delta 15). If Flat+Host still wins
        # here, the negative result about the graph stands on its own; if the
        # gap closes, the GNN's problem is wasted capacity -- a real criticism
        # but a different and more fixable one than "propagation does not
        # help".
        runs.append(
            _flat(
                "flathost_small_replay",
                "arch",
                (
                    "replay.enabled=true",
                    "model.use_host_features=true",
                    "model.mlp_hidden=123",
                ),
                seed,
                note="matched to the GNN's 156,562 *reachable* parameters, not its nominal count",
            )
        )

    # --- replay memory budget ------------------------------------------
    # Seed-outer so the whole curve exists at seed 42 before any second
    # seed starts: a 5-point shape is worth more mid-sweep than three
    # well-replicated points and two holes, and if the curve turns out flat
    # there is no reason to spend the remaining seeds on it.
    #
    # Run on BOTH encoders on purpose. The budget has to be chosen for
    # whichever model the write-up leads with, and the architecture ladder
    # currently has Flat+Host ahead of the GNN -- so picking a budget on the
    # GNN alone would risk fixing a number on the arm that does not ship.
    # It also tests something worth knowing in its own right: if the two
    # encoders want different amounts of memory, the budget is a property of
    # the model rather than of the benchmark, and neither number transfers.
    for seed in seeds:
        for budget in BUDGETS:
            runs.append(
                _gnn(
                    f"budget_gnn_{budget}",
                    "budget",
                    (
                        "replay.enabled=true",
                        *NO_EWC,
                        *FIXED_REHEARSAL,
                        f"replay.buffer_size_per_task={budget}",
                    ),
                    seed,
                    note=f"replay buffer of {budget} graphs per task",
                )
            )
    for seed in seeds:
        for budget in BUDGETS:
            runs.append(
                _flat(
                    f"budget_flathost_{budget}",
                    "budget",
                    (
                        "replay.enabled=true",
                        "model.use_host_features=true",
                        *FIXED_REHEARSAL,
                        f"replay.buffer_size_per_task={budget}",
                    ),
                    seed,
                    note=f"replay buffer of {budget} graphs per task",
                )
            )

    # --- continual-learning methods ------------------------------------
    for seed in seeds:
        runs.append(_gnn("gnn_finetune", "method", ("replay.enabled=false", *NO_EWC), seed))
        runs.append(
            _gnn(
                "gnn_ewc",
                "method",
                (
                    "replay.enabled=false",
                    "ewc.lambda_r=1",
                    "ewc.lambda_s=1",
                    "ewc.lambda_u=1",
                    *SHORT_WARMUP,
                ),
                seed,
                note="relation-aware Online EWC, the mechanism TRD replaces",
            )
        )
        # Doubles as the architecture ladder's top rung -- same config, so
        # running it twice under two names would only add noise.
        runs.append(
            _gnn("gnn_replay", "arch", ("replay.enabled=true", *NO_EWC), seed,
                 note="also the CL baseline TRD is measured against")
        )
        # Same config as gnn_replay with one change: concatenate the per-
        # relation embeddings instead of taking HAN's softmax-weighted
        # average. Added after the ladder came out backwards -- Flat+Host
        # (no message passing at all) beat the GNN 0.854 to 0.749 at matched
        # parameters -- to separate two explanations the ladder cannot:
        # "message passing does not help here" versus "message passing helps
        # but the convex-combination fusion threw the result away". The flat
        # arms concatenate; the GNN averaged. Costs +255,506 - 264,850 <
        # -3.5% nominal parameters (fewer, not more: ConcatFusion no longer
        # allocates a projection for the three node types with only one
        # incoming relation -- Protocol/Service/Port -- see
        # attention_fusion.ConcatFusion's docstring), far too little to
        # explain a 10-point gap either way. Reachable params are unchanged
        # at 168,658 either side of that fix, since it only removed weights
        # that were already never in the forward pass.
        runs.append(
            _gnn(
                "gnn_concat_replay",
                "arch",
                ("replay.enabled=true", *NO_EWC, "model.fusion=concat"),
                seed,
                note="fusion ablation: concat instead of HAN convex combination",
            )
        )
        # TRD is tested WITHOUT replay, matching gnn_ewc above. This is the
        # like-for-like comparison against the mechanism it replaces: EWC was
        # only ever evaluated in the no-replay regime, and testing TRD with
        # replay while EWC ran without it would compare the regimes rather
        # than the mechanisms.
        #
        # It is also the regime where a drift regulariser has room to act.
        # Replay already anchors the encoder by rehearsing old data, so a
        # penalty pulling toward the task t-1 state fights it rather than
        # complementing it -- measured, not assumed: at lambda_d=0.5 with
        # replay, TRD scored 0.707 accuracy / 0.226 forgetting against plain
        # replay's 0.764 / 0.202, with the penalty term substantial
        # (0.12-0.23 against a cross-entropy of ~0.4) rather than inert.
        # "drift" is the reformulation the drift.py measurement motivates:
        # transferability is anti-correlated with per-relation drift (r =
        # -0.909 on the unbiased 5-boundary numbers), so softmax(S_r)
        # concentrates the penalty where there is least to preserve, and
        # "inverse" -- which won the first weighting comparison -- was only
        # ever a proxy for weighting by D_r directly. Same live-measurement
        # cost as every other TRD arm plus two no-grad forward passes over
        # <=60 graphs per task.
        for weighting in ("uniform", "transfer", "inverse", "drift"):
            runs.append(
                _gnn(
                    f"gnn_trd_{weighting}",
                    "method",
                    (
                        "replay.enabled=false",
                        *NO_EWC,
                        *SHORT_WARMUP,
                        # Temperature must match the signal's scale, not be
                        # identical across modes -- see DRIFT_TEMPERATURE.
                        *(DRIFT_TEMPERATURE if weighting == "drift" else TEMPERATURE),
                        "distill.enabled=true",
                        f"distill.weighting={weighting}",
                        f"distill.lambda_d={lambda_d}",
                    ),
                    seed,
                )
            )
        # And one combined arm: does TRD add anything on top of the strongest
        # simple baseline, or is replay already doing that job? Only the
        # winning weighting is run combined -- the weighting question is
        # settled in the no-replay arms above, and three more arms here would
        # cost 4.5 hours to re-answer it under a confound.
        runs.append(
            _gnn(
                "gnn_replay_trd",
                "method",
                (
                    "replay.enabled=true",
                    *NO_EWC,
                    *SHORT_WARMUP,
                    *TEMPERATURE,
                    "distill.enabled=true",
                    "distill.weighting=transfer",
                    f"distill.lambda_d={lambda_d}",
                ),
                seed,
                note="replay + TRD: does the regulariser add on top of replay?",
            )
        )

    runs.append(
        Run(
            name="joint",
            module="trench_ids.cl.train_joint",
            overrides=("train.seed=42",),
            group="joint",
            note="non-continual upper bound",
        )
    )
    return runs


_ARM_SUFFIX_RE = re.compile(r"_s\d+$")


def arm_of(run_name: str) -> str:
    """The arm a run belongs to, with its ``_s<seed>`` suffix removed.

    ``--only`` matches on this rather than on a raw string prefix, because
    the arm names are not prefix-free: ``gnn_replay`` is a prefix of
    ``gnn_replay_trd``, so a prefix filter aimed at plain replay would also
    pull in the replay+TRD arm -- which depends on a selected lambda_d and
    would otherwise be run at the placeholder default, quietly writing a
    result that looks like the real thing.
    """
    return _ARM_SUFFIX_RE.sub("", run_name)


@dataclass
class Result:
    completed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def execute(runs: list[Run], dry_run: bool = False) -> Result:
    result = Result()
    for i, run in enumerate(runs, start=1):
        summary = run.summary_path()
        if summary.exists():
            print(f"[{i}/{len(runs)}] skip {run.name} (already finished)", flush=True)
            result.skipped.append(run.name)
            # A run finished by an earlier, interrupted sweep may still be
            # missing its pooled metrics; fill them in rather than leaving a
            # hole in the comparison table.
            if not (run.out_dir() / "eval" / "pooled_final_metrics.json").exists() and not dry_run:
                _evaluate(run)
            continue
        command = run.command()
        print(f"[{i}/{len(runs)}] run  {run.name}: {' '.join(command[2:])}", flush=True)
        if dry_run:
            continue
        start = time.monotonic()
        run.out_dir().mkdir(parents=True, exist_ok=True)
        # Per-run log on disk rather than only in this process' memory: a
        # multi-hour sweep has to be inspectable while it is still running,
        # and a failure has to leave evidence behind after the sweep moves on.
        log_path = run.out_dir() / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        elapsed = time.monotonic() - start
        if completed.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            print(f"    FAILED after {elapsed:.0f}s\n{tail}", flush=True)
            result.failed.append(run.name)
            continue
        print(f"    done in {elapsed / 60:.1f} min", flush=True)
        _evaluate(run)
        result.completed.append(run.name)
    return result


def _evaluate(run: Run) -> None:
    """Pooled precision/recall/F1 for a finished run.

    Run here rather than as a separate pass because ``compare_runs`` reads
    ``eval/pooled_final_metrics.json`` for the F1 columns, and F1-macro --
    not accuracy -- is the metric the comparison turns on: accuracy is
    dominated by Benign plus the two largest attack classes, so an arm can
    move it without touching the classes anyone cares about.
    """
    if run.module.endswith("train_joint"):
        # Joint training has no task sequence and writes no per-task
        # checkpoints, so the forgetting-matrix-based evaluator has nothing
        # to read. Its own summary already carries the upper-bound numbers.
        return
    command = [
        sys.executable,
        "-m",
        "trench_ids.cl.evaluate",
        "--run-dir",
        run.out_dir().as_posix(),
        "--out-dir",
        (run.out_dir() / "eval").as_posix(),
    ]
    log_path = run.out_dir() / "eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        print(f"    (evaluation failed; see {log_path})", flush=True)


def collect(runs: list[Run]) -> dict[str, dict]:
    """Gather every finished run's summary into one dict for reporting."""
    out: dict[str, dict] = {}
    for run in runs:
        summary = run.summary_path()
        if summary.exists():
            out[run.name] = {
                "group": run.group,
                "note": run.note,
                "overrides": list(run.overrides),
                **json.loads(summary.read_text()),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TRENCH-IDS experiment matrix.")
    parser.add_argument(
        "--group",
        default="all",
        help="lambda | arch | method | budget | joint | all (comma-separated for several)",
    )
    parser.add_argument(
        "--lambda-d",
        type=float,
        default=None,
        help=(
            "lambda_d for every TRD/Replay+TRD arm (the 'method' group). No safe "
            "default: the project's selected value is 0.25 (the 'lambda' group's "
            "val-accuracy selection, see docs/results.md), and 1.0 -- an "
            "arbitrary placeholder, not a considered default -- has already been "
            "measured worse (val 0.404 at 0.5 declining to 0.270 at 2.0). "
            "Required unless --group excludes 'method' (and 'all')."
        ),
    )
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Comma-separated run-name prefixes to restrict to. Lets several workers "
            "share the matrix without racing for the same run directory: the GPU sits "
            "near-idle because collation is CPU-bound, so the sweep runs several "
            "processes wide, and two workers picking up the same unfinished run would "
            "both write into it."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    wanted = None if args.group == "all" else set(args.group.split(","))
    needs_lambda_d = wanted is None or "method" in wanted
    if needs_lambda_d and args.lambda_d is None:
        raise SystemExit(
            "--lambda-d is required when running the 'method' group (or --group "
            "all): there is no safe default. Pass --lambda-d 0.25 (the project's "
            "val-selected value, docs/results.md) to reproduce the reported "
            "results, or restrict to --group lambda,arch,budget,joint to skip "
            "the TRD arms entirely."
        )
    runs = build_matrix(args.lambda_d if args.lambda_d is not None else float("nan"), seeds)
    if wanted is not None:
        runs = [r for r in runs if r.group in wanted]
    if args.only:
        wanted_arms = set(args.only.split(","))
        runs = [r for r in runs if arm_of(r.name) in wanted_arms]
        if not runs:
            raise SystemExit(f"--only={args.only} matched no runs in group {args.group}")

    if args.collect_only:
        print(json.dumps(collect(runs), indent=2))
        return

    result = execute(runs, dry_run=args.dry_run)
    print(
        f"[matrix] completed={len(result.completed)} "
        f"skipped={len(result.skipped)} failed={len(result.failed)}"
    )
    if result.failed:
        print(f"[matrix] failed runs: {result.failed}")


if __name__ == "__main__":
    main()
