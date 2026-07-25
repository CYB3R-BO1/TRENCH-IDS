"""Port-representation figures for the writeup (2026-07-24).

Reads data/processed/task_*.parquet directly -- no training/eval dependency.

Run: .venv/Scripts/python.exe scripts/make_port_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "figure.titleweight": "bold",
})

TASK_LABELS = [
    "T1 Scanning", "T2 Recon.", "T3 DDoS+Infil.",
    "T4 DoS+Inject.", "T5 Pwd+Bot", "T6 XSS+Brute",
]

PORT_SERVICE_NAMES = {
    80: "80/HTTP", 53: "53/DNS", 443: "443/HTTPS", 21: "21/FTP", 22: "22/SSH",
    993: "993/IMAPS", 25: "25/SMTP", 587: "587/SMTP-sub", 143: "143/IMAP",
    445: "445/SMB", 995: "995/POP3S", 135: "135/RPC", 110: "110/POP3",
    139: "139/NetBIOS", 554: "554/RTSP",
}


def load_task_dst_ports() -> dict[str, pd.Series]:
    per_task = {}
    for p in sorted((ROOT / "data" / "processed").glob("task_*.parquet")):
        task = p.stem.split("_")[1]
        df = pd.read_parquet(p, columns=["L4_DST_PORT"])
        per_task[task] = df["L4_DST_PORT"]
    return per_task


def plot_well_known_share_per_task(per_task: dict[str, pd.Series]) -> None:
    tasks = sorted(per_task, key=int)
    shares = [((per_task[t] >= 0) & (per_task[t] <= 1023)).mean() for t in tasks]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#C44E52" if s < 0.5 else "#55A868" for s in shares]
    ax.bar(range(len(tasks)), shares, color=colors)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(TASK_LABELS[: len(tasks)], rotation=30, ha="right")
    ax.set_ylabel("Share of flows with dst port in 0-1023")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.712, color="black", linestyle="--", alpha=0.5, label="overall mean (71.2%)")
    for i, s in enumerate(shares):
        ax.text(i, s + 0.02, f"{s:.1%}", ha="center", fontsize=9)
    ax.set_title("Well-known port (0-1023) share by task")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "port_wellknown_share_by_task.png", dpi=150)
    plt.close(fig)


def plot_top_ports_overall(per_task: dict[str, pd.Series]) -> None:
    all_ports = pd.concat(per_task.values())
    well_known = all_ports[(all_ports >= 0) & (all_ports <= 1023)]
    top = well_known.value_counts().head(12)
    labels = [PORT_SERVICE_NAMES.get(p, str(p)) for p in top.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top)), top.values[::-1], color="#4C72B0")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Flow count (log scale)")
    ax.set_xscale("log")
    ax.set_title("Top 12 well-known destination ports (all tasks pooled)\n"
                  "-- each now gets its own embedding row instead of sharing one bucket")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "port_top_wellknown_ports.png", dpi=150)
    plt.close(fig)


def plot_collision_before_after() -> None:
    """Illustrates the specific bug: ports 20/21/22/23 in the old 32-bucket
    log scheme vs. the new exact-index scheme."""
    import math

    def old_bucket(port: int, num_buckets: int = 32) -> int:
        scaled = math.log1p(port) / math.log1p(65535)
        return min(num_buckets - 1, round(scaled * (num_buckets - 1)))

    ports = [20, 21, 22, 23]
    names = ["20/FTP-data", "21/FTP-ctrl", "22/SSH", "23/Telnet"]
    old_idx = [old_bucket(p) for p in ports]
    new_idx = ports  # exact identity

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, old_idx, color="#C44E52")
    axes[0].set_title("Old scheme: log-bucket index\n(all collide into bucket 9)")
    axes[0].set_ylabel("Embedding row index")
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(old_idx):
        axes[0].text(i, v + 0.3, str(v), ha="center")

    axes[1].bar(names, new_idx, color="#55A868")
    axes[1].set_title("New scheme: exact port index\n(each gets its own row)")
    axes[1].set_ylabel("Embedding row index")
    axes[1].tick_params(axis="x", rotation=30)
    for i, v in enumerate(new_idx):
        axes[1].text(i, v + 0.3, str(v), ha="center")

    fig.suptitle("Port representation fix: FTP/SSH/Telnet collision")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "port_collision_before_after.png", dpi=150)
    plt.close(fig)


def main() -> None:
    per_task = load_task_dst_ports()
    plot_well_known_share_per_task(per_task)
    plot_top_ports_overall(per_task)
    plot_collision_before_after()
    print(f"[done] wrote 3 port figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
