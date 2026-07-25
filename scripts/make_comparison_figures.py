"""Generates comparison figures for the EWC-vs-replay writeup (2026-07-24).

Reads existing run outputs (runs/step4, runs/replay_baseline,
runs/hp_sweep/*, runs/joint_upper_bound) -- no training, pure plotting.

Run: .venv/Scripts/python.exe scripts/make_comparison_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 19,
    "figure.titleweight": "bold",
})

TASK_LABELS = [
    "T1\n(Scanning)", "T2\n(Recon.)", "T3\n(DDoS + Infil.)",
    "T4\n(DoS + Inject.)", "T5\n(Pwd + Bot)", "T6\n(XSS + Brute)",
]


def add_caption(fig, text: str) -> None:
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=9.5,
              color="#444444", style="italic", wrap=True)


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def forgetting_matrix_to_grid(fm: dict) -> np.ndarray:
    n = len(fm)
    grid = np.full((n, n), np.nan)
    for trained_up_to_str, row in fm.items():
        t = int(trained_up_to_str) - 1
        for evaluated_task_str, acc in row.items():
            e = int(evaluated_task_str) - 1
            grid[t, e] = acc
    return grid


def plot_forgetting_heatmap(fm: dict, title: str, out_name: str) -> None:
    grid = forgetting_matrix_to_grid(fm)
    n = grid.shape[0]
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(TASK_LABELS[:n], rotation=0, ha="center", fontsize=9.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"after T{i + 1}" for i in range(n)])
    ax.set_xlabel("Evaluated on task")
    ax.set_ylabel("Trained up to")
    ax.set_title(title, pad=12)
    for i in range(n):
        for j in range(n):
            if not np.isnan(grid[i, j]):
                color = "black" if 0.35 < grid[i, j] < 0.75 else "white"
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                         fontsize=11.5, color=color, fontweight="medium")
    fig.colorbar(im, ax=ax, label="test accuracy")
    fig.tight_layout()
    add_caption(fig, "Cell = test accuracy on the column's task, evaluated after training up through the row's task.")
    fig.savefig(OUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_method_bar_chart() -> None:
    """Replay's bars carry error bars from the clean 3-seed run (seeds
    42/1/2, original port scheme -- the mislabeled "seed 3" run is excluded,
    it was actually seed 2 rerun under the post-fix port embedding, kept
    separately as the controlled port-scheme comparison, not a 4th seed;
    see docs/port-representation-proposal.md). Macro F1 was recovered for
    all 3 via an isolated git worktree at the pre-port-fix commit, since
    seed 1/2's checkpoints can't be loaded by the current (post-fix)
    inference code (embedding table shape changed).
    """
    methods = ["EWC\n(baseline)", "EWC\n(LR 3e-4)", "Replay\n(3-seed)", "Joint"]
    single_seed = [True, True, False, True]
    accuracy = [0.36, 0.347, 0.8610, 0.876]
    accuracy_err = [0, 0, 0.0135, 0]
    macro_f1 = [0.166, 0.154, 0.8046, 0.842]
    macro_f1_err = [0, 0, 0.0097, 0]
    forgetting = [0.705, 0.710, 0.0939, np.nan]
    forgetting_err = [0, 0, 0.0151, 0]

    x = np.arange(len(methods))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
    bars_acc = ax1.bar(x - width, accuracy, width, yerr=accuracy_err, capsize=4,
                        label="Final accuracy", color="#4C72B0")
    bars_f1 = ax1.bar(x, macro_f1, width, yerr=macro_f1_err, capsize=4,
                       label="Macro F1", color="#55A868")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)

    ax2 = ax1.twinx()
    ax2.bar(x + width, forgetting, width, yerr=forgetting_err, capsize=4,
            label="Avg. forgetting", color="#C44E52")
    ax2.set_ylabel("Average forgetting")
    ax2.set_ylim(0, 1)

    for bars, values in [(bars_acc, accuracy), (bars_f1, macro_f1)]:
        for bar, val, is_single in zip(bars, values, single_seed):
            if is_single:
                ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.015, "*",
                          ha="center", va="bottom", fontsize=14, color="#333333")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.18))
    fig.suptitle("Method Comparison", y=1.11, fontsize=19)
    fig.text(0.5, 1.015, "Accuracy, Macro-F1 and Average Forgetting", ha="center",
              fontsize=13, color="#444444")
    fig.tight_layout()
    add_caption(
        fig,
        "Joint training serves as the upper-bound reference (all tasks trained jointly, no continual-learning "
        "setting). Replay: mean ± std over 3 seeds. * single-seed evaluation, no variance estimate yet."
    )
    fig.savefig(OUT_DIR / "method_comparison_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_over_tasks(ewc_fm: dict, replay_fm: dict) -> None:
    n = len(ewc_fm)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    for task_idx in range(n):
        ax = axes[task_idx]
        ewc_curve = []
        replay_curve = []
        xs = list(range(task_idx + 1, n + 1))
        for trained_up_to in xs:
            ewc_curve.append(ewc_fm[str(trained_up_to)][str(task_idx + 1)])
            replay_curve.append(replay_fm[str(trained_up_to)][str(task_idx + 1)])
        ax.plot(xs, ewc_curve, "o-", color="#C44E52", label="EWC")
        ax.plot(xs, replay_curve, "o-", color="#55A868", label="Replay")
        ax.set_title(TASK_LABELS[task_idx])
        ax.set_xlabel("trained up to task")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if task_idx == 0:
            ax.set_ylabel("test accuracy")
            ax.legend()
    fig.suptitle("Per-Task Accuracy as Training Proceeds")
    fig.tight_layout()
    add_caption(fig, "Each panel tracks one task's test accuracy as the model continues training on later tasks.")
    fig.savefig(OUT_DIR / "accuracy_over_tasks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forgetting_per_task(ewc_fm: dict, replay_fm: dict) -> None:
    n = len(ewc_fm)

    def per_task_forgetting(fm: dict) -> list[float]:
        final_task = n
        values = []
        for evaluated_task in range(1, final_task):
            peak = max(
                fm[str(trained_up_to)][str(evaluated_task)]
                for trained_up_to in range(evaluated_task, final_task)
            )
            final_acc = fm[str(final_task)][str(evaluated_task)]
            values.append(peak - final_acc)
        return values

    ewc_vals = per_task_forgetting(ewc_fm)
    replay_vals = per_task_forgetting(replay_fm)
    x = np.arange(len(ewc_vals))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, ewc_vals, width, label="EWC", color="#C44E52")
    ax.bar(x + width / 2, replay_vals, width, label="Replay", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS[: len(ewc_vals)], rotation=30, ha="right")
    ax.set_ylabel("Forgetting (peak - final accuracy)")
    ax.set_title("Per-Task Forgetting", pad=12)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    add_caption(fig, "Excludes the final task (T6), which has no later task to be forgotten against.")
    fig.savefig(OUT_DIR / "forgetting_per_task.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_replay_multiseed() -> None:
    """Replay reproducibility across 3 seeds (42/1/2, original port scheme)
    only -- kept separate from the port-scheme comparison (a different
    experimental question, see plot_port_scheme_per_task_comparison), per
    the reviewer note that mixing the two stories in one figure is confusing."""
    seeds = ["seed 42", "seed 1", "seed 2"]
    forgetting = [0.0950, 0.1084, 0.0783]
    accuracy = [0.8571, 0.8498, 0.8760]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

    for ax, vals, ylabel, title in [
        (axes[0], forgetting, "Avg. forgetting", "Forgetting"),
        (axes[1], accuracy, "Final avg. accuracy", "Accuracy"),
    ]:
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)
        ax.scatter(range(3), vals, color="#55A868", s=70, zorder=3, label="individual seeds")
        ax.axhline(mean, color="#55A868", linestyle="--", alpha=0.6,
                   label=f"mean = {mean:.3f} ± {std:.3f}")
        ax.set_xticks(range(3))
        ax.set_xticklabels(seeds)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Replay Reproducibility (3 seeds)")
    fig.tight_layout()
    add_caption(fig, "All 3 seeds use the original (pre-fix) port representation scheme.")
    fig.savefig(OUT_DIR / "replay_multiseed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_port_scheme_per_task_comparison() -> None:
    """Same-seed (seed=2) old-vs-new port scheme comparison, per task, final
    (post-task-6) accuracy -- runs/replay_seed2 vs runs/replay_seed2_newport.
    Highlights the T3/T4 regression under the new scheme despite T1/T2/T5/T6
    holding steady or improving slightly."""
    old_fm = load("runs/replay_seed2/forgetting_matrix.json")
    new_fm = load("runs/replay_seed2_newport/forgetting_matrix.json")
    n = len(old_fm)
    old_final = [old_fm[str(n)][str(t)] for t in range(1, n + 1)]
    new_final = [new_fm[str(n)][str(t)] for t in range(1, n + 1)]

    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, old_final, width, label="Old port scheme (log-bucket)", color="#4C72B0")
    ax.bar(x + width / 2, new_final, width, label="New port scheme (exact 0-1023)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_LABELS[:n], rotation=30, ha="right")
    ax.set_ylabel("Final accuracy (after task 6)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Effect of Port Representation on Forgetting", pad=12)
    for i, (o, nv) in enumerate(zip(old_final, new_final)):
        if abs(nv - o) > 0.05:
            ax.annotate(f"{nv - o:+.2f}", (i, max(o, nv) + 0.03), ha="center", fontsize=10,
                         color="#C44E52", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    add_caption(
        fig,
        "Same seed (2), old (log-bucketed) vs. new (exact well-known-port) representation. "
        "T3/T4 regress under the new scheme; T1/T2/T5/T6 hold steady or improve. Single seed pair, not yet multi-seed."
    )
    fig.savefig(OUT_DIR / "port_scheme_per_task_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ewc_fm = load("runs/step4/forgetting_matrix.json")
    replay_fm = load("runs/replay_baseline/forgetting_matrix.json")

    plot_forgetting_heatmap(ewc_fm, "Continual Learning Performance (Online EWC)", "forgetting_heatmap_ewc.png")
    plot_forgetting_heatmap(replay_fm, "Continual Learning Performance (Replay)", "forgetting_heatmap_replay.png")
    plot_method_bar_chart()
    plot_accuracy_over_tasks(ewc_fm, replay_fm)
    plot_forgetting_per_task(ewc_fm, replay_fm)
    plot_replay_multiseed()
    plot_port_scheme_per_task_comparison()

    print(f"[done] wrote 7 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
