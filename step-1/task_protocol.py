"""Build Step 1 allocation and feasibility evidence without rerunning the sweeps."""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import groupby
import json
from pathlib import Path

import numpy as np
import pandas as pd

import attack_class_similarity as acs
from window_sweep import load_stream, split_of_windows, window_base, window_counts


STEP = Path(__file__).resolve().parent
TIME = "FLOW_START_MILLISECONDS"
WINDOW_COLUMNS = ["window_id", "split", "class", "attack_flows", "benign_available"]
THRESHOLDS = (5, 10, 20)
RELATION_FIELDS = ("L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def seeded_order(classes, seed):
    return sorted(classes, key=lambda name: (
        hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(), name))


def assign_benign(windows, eligible):
    """Allocate same-window benign once, retaining each present class's attack rows."""
    if windows.duplicated(["window_id", "class"]).any():
        raise ValueError("duplicate class/window")
    if (windows.groupby("window_id")[["split", "benign_available"]].nunique() > 1).any().any():
        raise ValueError("physical window has conflicting splits or benign counts")
    if (windows["attack_flows"] <= 0).any() or (windows["benign_available"] < 0).any():
        raise ValueError("expected attack-present windows with nonnegative benign counts")

    result = windows.loc[windows["class"].isin(eligible), WINDOW_COLUMNS].copy()
    result["support"] = result.groupby(["split", "class"])["window_id"].transform("size")
    result = result.sort_values(["window_id", "class"]).reset_index(drop=True)
    candidates = result.loc[result["benign_available"] > 0,
                            ["window_id", "split", "class", "attack_flows", "support"]]
    awards = Counter()
    owners = []

    def priority(row):
        _, _, split, name, attacks, support = row
        return Fraction(awards[(split, name)], support), support, -attacks, name

    for _, window in groupby(candidates.itertuples(name=None), key=lambda row: row[1]):
        index, _, split, name, _, _ = min(window, key=priority)
        awards[(split, name)] += 1
        owners.append(index)
    result["benign_owner"] = result.index.isin(owners)
    result["benign_flows"] = result["benign_available"].where(result["benign_owner"], 0)
    return result.drop(columns="support")


def summarize(allocation, eligible):
    rows = []
    for name in sorted(eligible):
        for split in acs.SPLIT_NAMES:
            view = allocation[(allocation["class"] == name) & (allocation["split"] == split)]
            attacks = view["attack_flows"]
            rows.append({
                "class": name, "split": split, "usable_windows": len(view),
                **{f"attack_ge{n}": int((attacks >= n).sum()) for n in THRESHOLDS},
                "median_attack": attacks.median(),
                "attack_flows": int(attacks.sum()),
                "benign_windows": int(view["benign_owner"].sum()),
                "benign_flows": int(view["benign_flows"].sum()),
            })
    return pd.DataFrame(rows)


def raw_unsw_windows(path, manifest, size, source):
    labels = source["attack_classes"]
    timestamps, codes, code_of = load_stream(path, {"chunk": 250_000}, labels)
    if len(timestamps) != source["n_rows"]:
        raise ValueError("UNSW row count differs from frozen sweep")
    base = window_base(timestamps, manifest["align_ms"])
    total, benign, counts = window_counts(timestamps, codes, len(labels), size, base)
    cuts = manifest["cuts"]
    splits = np.asarray(acs.SPLIT_NAMES)[
        split_of_windows(len(total), size, cuts["t70"], cuts["t85"], base)
    ]
    for split, expected in manifest["per_window"][str(size)].items():
        mask = splits == split
        if (total[mask].sum(), benign[mask].sum(), (total[mask] > 0).sum()) != (
                expected["flows"], expected["benign_flows"], expected["windows"]):
            raise ValueError(f"UNSW window totals differ from frozen sweep for {split}")
    frames = []
    for name in labels:
        attack = counts[code_of[name]]
        present = np.flatnonzero(attack)
        frames.append(pd.DataFrame({
            "window_id": present + base // size, "split": splits[present], "class": name,
            "attack_flows": attack[present], "benign_available": benign[present],
        }))
    return pd.concat(frames, ignore_index=True)


def stored_ton_windows(size, cuts):
    columns = ["window_ms", "window_id", "population", "class", "split", "n_attack", "n_benign"]
    path = STEP / f"outputs/ton_uncertainty/temporal_relations/relation_window_statistics_{size // 1000}s.csv"
    source = pd.read_csv(path, usecols=columns)
    windows = source.loc[source["population"] == "A"].rename(
        columns={"n_attack": "attack_flows", "n_benign": "benign_available"})
    if not windows["window_ms"].eq(size).all() or windows["window_id"].duplicated().any():
        raise ValueError("ToN evidence contains mis-sized or shared attack-present windows")
    for name, group in windows.groupby("class"):
        start = group["window_id"] * size
        limits = cuts[name]
        assigned = np.select(
            [start < limits["t70"], start < limits["t85"]], ["train", "val"], default="test")
        if not group["split"].eq(assigned).all():
            raise ValueError(f"ToN stored split mismatch for {name}")
    return windows[WINDOW_COLUMNS]


def verify_attack_counts(windows, path, size):
    frozen = pd.read_csv(path).rename(columns={
        "median_class_flows": "median_attack", "median_flows_per_class_window": "median_attack",
        "nonempty_windows": "usable_windows", "class_flows": "attack_flows",
        **{f"windows_class_ge{n}": f"attack_ge{n}" for n in THRESHOLDS},
    })
    frozen = frozen.loc[frozen["window_ms"] == size].set_index(["class", "split"])
    observed = summarize(assign_benign(windows, set(windows["class"])),
                         set(windows["class"])).set_index(["class", "split"])
    if not observed.index.sort_values().equals(frozen.index.sort_values()):
        raise ValueError(f"classes or splits differ from {path}")
    fields = ["usable_windows", "attack_flows", "median_attack",
              *(f"attack_ge{n}" for n in THRESHOLDS)]
    # Legacy sweeps record a median of 0 for empty groups.
    pd.testing.assert_frame_equal(observed[fields].sort_index().fillna(0),
                                  frozen[fields].sort_index(), check_dtype=False)


def recompute_ton_relation_pair(path, cuts, names, expected_counts):
    """Compute only the seven missing canonical blocks for the two specified classes."""
    cols = (acs.HOST_SRC, acs.HOST_DST, "host_pair", *RELATION_FIELDS)
    counts = {name: {col: Counter() for col in cols} for name in names}
    observed_counts = Counter()
    usecols = ["Attack", TIME, acs.HOST_SRC, acs.HOST_DST, *RELATION_FIELDS]
    for chunk in acs._iter_csv(path, {"chunk": 250_000}, usecols=usecols):
        for name in names:
            rows = chunk.loc[chunk["Attack"].eq(name)]
            times = acs.validate_timestamp_column(rows[TIME].to_numpy(dtype=np.float64), TIME)
            rows = rows.loc[times < cuts[name]["t70"]]
            observed_counts[name] += len(rows)
            src, dst = rows[acs.HOST_SRC], rows[acs.HOST_DST]
            counts[name][acs.HOST_SRC].update(src)
            counts[name][acs.HOST_DST].update(dst)
            counts[name]["host_pair"].update(zip(src, dst))
            for col in RELATION_FIELDS:
                valid, codes, _ = acs.categorical_field_codes(col, rows[col].to_numpy(dtype=float))
                counts[name][col].update(codes[valid])
    if dict(observed_counts) != expected_counts:
        raise ValueError("ToN relation training support differs from frozen sweep")
    distances = {}
    for block, col in zip(acs.RELATION_BLOCKS, cols, strict=True):
        left, right = (counts[name][col] for name in names)
        if not left or not right:
            raise ValueError(f"undefined relation {block}")
        if col in RELATION_FIELDS:
            universe = sorted(left.keys() | right.keys())
            a, b = (acs._prob_vector(ct, universe) for ct in (left, right))
        else:
            length = max(len(left), len(right))
            a, b = (acs.sorted_share_distribution(ct, length) for ct in (left, right))
        distances[block] = acs._js(a, b)
    return distances


def stored_distances(path, names):
    frame = pd.read_csv(path)
    mask = (frame["class_a"].isin(names) & frame["class_b"].isin(names)
            & frame["class_a"].ne(frame["class_b"]) & frame["defined"])
    distances = frame.loc[mask].set_index("block")["js_distance"].to_dict()
    if distances.keys() != set(acs.RELATION_BLOCKS):
        raise ValueError(f"missing relation blocks in {path}")
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write-allocations", action="store_true",
                        help="also export per-window ownership for downstream graph construction")
    parser.add_argument("--recompute-ton-relations", action="store_true",
                        help="scan the ToN CSV once for the missing canonical two-class distances")
    args = parser.parse_args()
    protocol = read_json(STEP / "step1_protocol.json")
    size = protocol["window"]["primary_ms"]
    args.out.mkdir(parents=True, exist_ok=True)

    results = {}
    for dataset, cfg in protocol["datasets"].items():
        print(f"Checking {dataset} {size // 1000} s windows", flush=True)
        manifest_path = STEP / cfg["splits"]["graph_manifest"]
        manifest = read_json(manifest_path)
        data = STEP.parent / acs.DATASETS[dataset][0]
        if dataset == "unsw":
            run = read_json(manifest_path.with_name("run_manifest.json"))
            windows = raw_unsw_windows(data, manifest, size, run)
        else:
            cuts = manifest["per_class_cuts"]
            baseline = read_json(STEP / cfg["splits"]["similarity_manifest"])
            if cuts != baseline["per_class_cuts"]:
                raise ValueError("ToN graph and normalization split cuts differ")
            if baseline["benign_audit"]["benign_flows_by_split"]["train"] != cfg["splits"]["normalization_benign_train_flows"]:
                raise ValueError("ToN corrected benign normalization reference changed")
            windows = stored_ton_windows(size, cuts)
        verify_attack_counts(windows, manifest_path.with_name("attack_class_window_statistics.csv"), size)
        allocated = assign_benign(windows, cfg["eligible_classes"])
        task_stats = summarize(allocated, cfg["eligible_classes"])
        if task_stats[["attack_flows", "benign_flows"]].eq(0).any().any():
            raise ValueError(f"{dataset} has tasks with undefined binary metrics")
        results[dataset] = task_stats.assign(dataset=dataset, window_ms=size)
        if args.write_allocations:
            allocated.to_csv(args.out / f"{dataset}_window_allocation_{size}ms.csv", index=False)

    pd.concat(results.values(), ignore_index=True).to_csv(
        args.out / "task_feasibility_10s.csv", index=False)

    if args.recompute_ton_relations:
        print("Computing only the canonical ToN injection/ddos relation blocks", flush=True)
        ton = protocol["datasets"]["ton"]
        unsw = protocol["datasets"]["unsw"]
        names = next(pair["classes"] for pair in ton["pairs"] if pair["condition"] == "relation")
        unsw_names = next(pair["classes"] for pair in unsw["pairs"] if pair["condition"] == "relation")
        cuts = read_json(STEP / ton["splits"]["graph_manifest"])["per_class_cuts"]
        stats = results["ton"]
        expected = stats.loc[(stats["split"] == "train") & stats["class"].isin(names)]
        canonical = recompute_ton_relation_pair(
            STEP.parent / acs.DATASETS["ton"][0], cuts, names,
            expected.set_index("class")["attack_flows"].to_dict())
        expected_similarity = baseline_relation(ton, names)
        np.testing.assert_allclose(1 - np.mean(list(canonical.values())), expected_similarity, atol=1e-6, rtol=0)
        relation_rows = []
        for label, pair, values in [
            ("unsw_global_train", unsw_names, stored_distances(
                STEP / "outputs/unsw_full/relation_block_distance.csv", unsw_names)),
            ("ton_per_class_train", names, canonical),
            ("ton_global_train_sensitivity", names, stored_distances(
                STEP / "outputs/ton_global_chrono/relation_block_distance.csv", names)),
        ]:
            relation_rows.extend({"scope": label, "class_a": pair[0], "class_b": pair[1],
                                  "relation": block, "js_distance": values[block]}
                                 for block in acs.RELATION_BLOCKS)
        pd.DataFrame(relation_rows).to_csv(args.out / "relation_distances.csv", index=False, float_format="%.6f")


def baseline_relation(config, names):
    manifest = STEP / config["splits"]["similarity_manifest"]
    results = read_json(manifest.with_name("similarity_results.json"))
    return results["relation_profile"][names[0]][names[1]]


if __name__ == "__main__":
    main()
