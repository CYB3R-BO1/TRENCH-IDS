"""Per-window feasibility for graph-based continual-learning tasks.

The global split cuts are aligned to window boundaries by the caller, so a
physical graph window belongs to exactly one partition. Mixing statistics are keyed
by ``(split, window_id)``; validation/test classes therefore cannot influence
training-window mixing.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def _concat(parts: list) -> np.ndarray:
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts)


def window_feasibility(ts_parts: list, class_names: list, cuts: tuple[int, int],
                       window_ms: int, min_graph_flows: int = 2) -> pd.DataFrame:
    """Per (class, split) window occupancy and class-mixing statistics.

    ``class_windows_ge_min`` counts windows with >= min_graph_flows of *this class*.
    ``graph_windows_ge_min_total_flows`` counts windows with >= min_graph_flows of
    *total* flows (including benign), i.e. windows large enough to build a graph.
    """
    t70, t85 = cuts

    attack_presence: dict[tuple[int, int], set[int]] = defaultdict(set)
    total_flows: dict[tuple[int, int], int] = defaultdict(int)
    class_counts: dict[tuple[int, int, int], int] = {}

    for ci, name in enumerate(class_names):
        ts = _concat(ts_parts[ci]) if ci < len(ts_parts) else np.array([], np.int64)
        if ts.size == 0:
            continue
        split = np.where(ts < t70, 0, np.where(ts < t85, 1, 2))
        win = (ts // window_ms).astype(np.int64)
        for s in range(3):
            m = split == s
            if not m.any():
                continue
            uniq, cnt = np.unique(win[m], return_counts=True)
            for w, c in zip(uniq.tolist(), cnt.tolist()):
                key = (s, int(w))
                total_flows[key] += int(c)
                class_counts[(ci, s, int(w))] = int(c)
                if ci > 0:  # attack classes only, for mixing
                    attack_presence[key].add(ci)

    rows = []
    for ci, name in enumerate(class_names):
        ts = _concat(ts_parts[ci]) if ci < len(ts_parts) else np.array([], np.int64)
        if ts.size == 0:
            continue
        split = np.where(ts < t70, 0, np.where(ts < t85, 1, 2))
        win = (ts // window_ms).astype(np.int64)
        for s, sname in enumerate(SPLIT_NAMES):
            m = split == s
            if not m.any():
                rows.append({"class": name, "split": sname, "flows": 0, "nonempty_windows": 0,
                             "class_windows_ge_min": 0,
                             "graph_windows_ge_min_total_flows": 0,
                             "median_flows_per_window": 0, "max_flows_per_window": 0,
                             "mixed_windows": None, "single_class_windows": None})
                continue
            uniq, cnt = np.unique(win[m], return_counts=True)
            key = (s, np.asarray(uniq, dtype=np.int64))
            if ci > 0:
                mixed = int(sum(1 for w in key[1].tolist()
                                if len(attack_presence[(s, int(w))]) > 1))
                single = int(sum(1 for w in key[1].tolist()
                                 if len(attack_presence[(s, int(w))]) == 1))
            else:
                mixed = single = None
            total_ok = int(sum(1 for w in key[1].tolist()
                               if total_flows[(s, int(w))] >= min_graph_flows))
            rows.append({
                "class": name, "split": sname, "flows": int(m.sum()),
                "nonempty_windows": int(uniq.size),
                "class_windows_ge_min": int((cnt >= min_graph_flows).sum()),
                "graph_windows_ge_min_total_flows": total_ok,
                "median_flows_per_window": int(np.median(cnt)),
                "max_flows_per_window": int(cnt.max()),
                "mixed_windows": mixed,
                "single_class_windows": single,
            })
    return pd.DataFrame(rows)
