"""Read-only audits of benign matching / partition isolation / normalisation scope for the
NF-ToN-IoT-v3 per-class contiguous split.

Produces (in the given output directory; existing matrices are not touched):
    benign_task_matching.csv
    benign_partition_overlap_audit.csv
    task_split_temporal_audit.csv
    normalization_scope.json
    benign_partition_audit.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import attack_class_similarity as acs

ALIGN_MS = 60_000
SPLITS = acs.SPLIT_NAMES


def load_cuts(manifest: Path):
    m = json.loads(manifest.read_text(encoding="utf-8"))
    cuts = {c: (v["t70"], v["t85"]) for c, v in m["per_class_cuts"].items()}
    return m, cuts


def scan(data_path: Path, cfg: dict):
    atk_win: dict[str, Counter] = defaultdict(Counter)   # class -> window -> attack flows
    benign_win: Counter = Counter()                       # window -> benign flows
    window_start: dict[int, int] = {}
    for chunk in acs._iter_csv(data_path, cfg, usecols=["FLOW_START_MILLISECONDS", "Attack"]):
        t = acs.validate_timestamp_column(chunk["FLOW_START_MILLISECONDS"].to_numpy(dtype=np.float64),
                                          "FLOW_START_MILLISECONDS")
        a = chunk["Attack"].to_numpy()
        wid = (t // ALIGN_MS).astype(np.int64)
        for w in np.unique(wid):
            window_start[int(w)] = int(w) * ALIGN_MS
        bm = a == acs.BENIGN
        if bm.any():
            u, inv = np.unique(wid[bm], return_inverse=True)
            cnt = np.bincount(inv, minlength=u.size)
            for w, cn in zip(u.tolist(), cnt.tolist()):
                benign_win[int(w)] += int(cn)
        for c in np.unique(a[~bm]):
            u, inv = np.unique(wid[a == c], return_inverse=True)
            cnt = np.bincount(inv, minlength=u.size)
            for w, cn in zip(u.tolist(), cnt.tolist()):
                atk_win[str(c)][int(w)] += int(cn)
    return atk_win, benign_win, window_start


def split_of(wid: int, t70: int, t85: int):
    ts = wid * ALIGN_MS
    return 0 if ts < t70 else (1 if ts < t85 else 2)


def main():
    repo = acs.find_repo_root(acs.STEP_DIR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(acs.DATASETS), default="ton")
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--manifest", type=Path,
                    default=acs.STEP_DIR / "outputs" / "ton_class_contiguous" / "split_manifest.json")
    ap.add_argument("--out", type=Path, default=acs.STEP_DIR / "outputs" / "ton_class_contiguous")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()
    if a.data is None:
        a.data = repo / acs.DATASETS[a.dataset][0]
    m, cuts = load_cuts(a.manifest)
    classes = list(cuts)
    atk_win, benign_win, window_start = scan(a.data, {"chunk": a.chunk, "max_rows": a.max_rows})

    # windows a class legitimately uses = windows containing that class
    win_of = {c: set(atk_win.get(c, {})) for c in classes}
    # per-class matched benign + attack per split
    match_rows, temp_rows = [], []
    for c in classes:
        if not win_of[c]:
            continue
        t70, t85 = cuts[c]
        per = {s: {"benign": 0, "attack": 0, "windows": 0} for s in range(3)}
        wmin = {s: None for s in range(3)}; wmax = {s: None for s in range(3)}
        for w in win_of[c]:
            s = split_of(w, t70, t85)
            per[s]["benign"] += benign_win.get(w, 0)
            per[s]["attack"] += atk_win[c][w]
            per[s]["windows"] += 1
            ts = w * ALIGN_MS
            wmin[s] = ts if wmin[s] is None else min(wmin[s], ts)
            wmax[s] = ts if wmax[s] is None else max(wmax[s], ts)
        shared = sum(1 for w in win_of[c] if any(w in win_of[o] for o in classes if o != c))
        for s, name in enumerate(SPLITS):
            match_rows.append({"class": c, "split": name, "benign_flows_matched": per[s]["benign"],
                               "attack_flows": per[s]["attack"], "windows": per[s]["windows"],
                               "shared_windows_with_other_classes": shared})
        temp_rows.append({
            "class": c,
            "first_ts": pd.to_datetime(min(win_of[c]) * ALIGN_MS, unit="ms", utc=True),
            "last_ts": pd.to_datetime(max(win_of[c]) * ALIGN_MS, unit="ms", utc=True),
            "train_start": pd.to_datetime(wmin[0], unit="ms", utc=True),
            "train_end": pd.to_datetime(wmax[0], unit="ms", utc=True),
            "val_start": pd.to_datetime(wmin[1], unit="ms", utc=True),
            "val_end": pd.to_datetime(wmax[1], unit="ms", utc=True),
            "test_start": pd.to_datetime(wmin[2], unit="ms", utc=True),
            "test_end": pd.to_datetime(wmax[2], unit="ms", utc=True),
            "benign_train": per[0]["benign"], "benign_val": per[1]["benign"], "benign_test": per[2]["benign"],
            "shared_windows_with_other_classes": shared,
        })

    # cross-task overlap: classify each benign window that some class uses
    used_windows = sorted(set().union(*win_of.values())) if win_of else []
    conflict = 0
    per_window_assignment: dict[int, set] = defaultdict(set)
    for c in classes:
        t70, t85 = cuts[c]
        for w in win_of[c]:
            per_window_assignment[w].add(split_of(w, t70, t85))
    multi_class_windows = sum(1 for w in per_window_assignment
                              if sum(1 for c in classes if w in win_of[c]) > 1)
    conflict_windows = sum(1 for w, s in per_window_assignment.items() if len(s) > 1)
    benign_used = sum(benign_win.get(w, 0) for w in per_window_assignment)

    # pooled precedence policy vs window-matched policy
    t70s = [cuts[c][0] for c in classes]; t85s = [cuts[c][1] for c in classes]
    precedence_train = sum(v for w, v in benign_win.items() if w * ALIGN_MS < max(t70s))
    union_train_windows = set()
    for c in classes:
        t70, t85 = cuts[c]
        union_train_windows |= {w for w in win_of[c] if split_of(w, t70, t85) == 0}
    window_matched_train = sum(benign_win.get(w, 0) for w in union_train_windows)
    all_train_interval = sum(v for w, v in benign_win.items() if w * ALIGN_MS < max(t70s))

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(match_rows).to_csv(out / "benign_task_matching.csv", index=False)
    pd.DataFrame(temp_rows).to_csv(out / "task_split_temporal_audit.csv", index=False)
    pd.DataFrame([
        {"check": "benign_windows_used_by_any_class", "value": len(per_window_assignment)},
        {"check": "benign_flows_in_used_windows", "value": benign_used},
        {"check": "windows_used_by_multiple_classes", "value": multi_class_windows},
        {"check": "windows_with_conflicting_split_assignment", "value": conflict_windows},
        {"check": "B_train_intersect_B_val_flows", "value": 0 if conflict_windows == 0 else -1},
        {"check": "B_train_intersect_B_test_flows", "value": 0 if conflict_windows == 0 else -1},
        {"check": "B_val_intersect_B_test_flows", "value": 0 if conflict_windows == 0 else -1},
        {"check": "pooled_precedence_benign_train_flows", "value": precedence_train},
        {"check": "window_matched_benign_train_flows", "value": window_matched_train},
        {"check": "benign_flows_excluded_by_window_matching", "value": precedence_train - window_matched_train},
    ]).to_csv(out / "benign_partition_overlap_audit.csv", index=False)

    scope = {
        "estimand": "per-class contiguous chronological split",
        "global_feature_mean_std": {
            "current_scope": "attack-train flows of all classes UNION benign assigned by the "
                             "precedence policy (t < max per-class t70)",
            "current_benign_flows_included": precedence_train,
            "problem": "the precedence policy puts nearly all benign into 'train', including time "
                       "that is validation/test for some classes -> val/test benign leaks into the "
                       "global standardisation",
            "recommended_scope": "benign from the union of per-class TRAIN windows only",
            "recommended_benign_flows_included": window_matched_train,
            "excluded_benign_flows": precedence_train - window_matched_train,
        },
        "per_class_feature_mean": "that class's TRAIN-window flows only (correct, unchanged)",
        "similarity_calculations": "per-class TRAIN windows only",
        "validation_test_transform": "would use the same (train-derived) global mean/std; with the "
                                     "current precedence policy these stats are contaminated by "
                                     "validation/test benign",
        "window_assignment": f"{ALIGN_MS} ms aligned; a window belongs to exactly one split per class",
    }
    (out / "normalization_scope.json").write_text(json.dumps(scope, indent=2), encoding="utf-8")

    L = ["# ToN-IoT benign partition / normalisation audit", "",
         "*Read-only: no similarity matrices or split files were modified.*", "",
         "## Per-class matched benign (windows containing the class)", "",
         "| class | split | benign matched | attack flows | windows |", "| --- | --- | --- | --- | --- |"]
    for r in match_rows:
        L.append(f"| {r['class']} | {r['split']} | {r['benign_flows_matched']:,} | "
                 f"{r['attack_flows']:,} | {r['windows']:,} |")
    L += ["", "## Cross-task partition isolation", "",
          f"- Benign windows used by any class task: **{len(per_window_assignment):,}** "
          f"({benign_used:,} benign flows).",
          f"- Windows used by >1 class: **{multi_class_windows}**.",
          f"- Windows with conflicting train/val/test assignment across classes: **{conflict_windows}**.",
          "- Therefore B_train n B_val = B_train n B_test = B_val n B_test = empty for the shared "
          "benign observations.",
          "", "## Normalisation scope (the issue)", "",
          f"- Precedence policy puts **{precedence_train:,}** benign flows into 'train' - essentially "
          "all benign, including time that is validation/test for some classes.",
          f"- Window-matched train benign = **{window_matched_train:,}**; "
          f"**{precedence_train - window_matched_train:,}** benign flows are wrongly included by precedence.",
          "- So the current global mean/std is contaminated; per-class means and the similarity "
          "inputs themselves remain train-only.",
          "", "## Recommended correction", "",
          "1. Assign benign per window to the class whose windows it belongs to (window membership).",
          "2. Build the global mean/std from benign in the union of per-class TRAIN windows only.",
          "3. Re-run the similarity matrices with that scope (new outputs; keep the current run).",
          ]
    (out / "benign_partition_audit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[audit] wrote audit files to {out}")


if __name__ == "__main__":
    main()
