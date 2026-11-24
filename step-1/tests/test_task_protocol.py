import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

import attack_class_similarity as acs
from task_protocol import (
    RELATION_FIELDS, STEP, TIME, WINDOW_COLUMNS, assign_benign, read_json,
    recompute_ton_relation_pair, seeded_order, summarize,
)


class TaskProtocolTests(unittest.TestCase):
    def windows(self, rows):
        return pd.DataFrame(rows, columns=WINDOW_COLUMNS)

    def test_reversal_and_shuffled_input_leave_assignments_identical(self):
        rows = self.windows([
            (1, "train", "A", 10, 17), (1, "train", "B", 4, 17),
            (2, "train", "A", 7, 3), (2, "train", "B", 6, 3),
            (3, "train", "B", 8, 7),
            (3, "train", "supplementary", 30, 7),
            (4, "val", "A", 3, 0),
        ])
        forward = assign_benign(rows, ["A", "B"])
        reverse = assign_benign(rows.sample(frac=1, random_state=2), ["B", "A"])
        pd.testing.assert_frame_equal(forward, reverse)
        owners = forward.loc[forward["benign_owner"]].set_index("window_id")
        self.assertEqual(owners["class"].to_dict(), {1: "A", 2: "B", 3: "B"})
        self.assertEqual(forward["benign_flows"].sum(), 27)
        self.assertEqual(len(forward), 6)
        self.assertEqual(forward["attack_flows"].sum(), 38)

    def test_volume_then_label_break_ties_and_each_split_starts_fresh(self):
        rows = self.windows([
            (1, "train", "A", 4, 7), (1, "train", "B", 5, 7),
            (2, "val", "A", 6, 8), (2, "val", "B", 6, 8),
        ])
        result = assign_benign(rows, ["B", "A"])
        owners = result.loc[result["benign_owner"]].set_index("window_id")
        self.assertEqual(owners["class"].to_dict(), {1: "B", 2: "A"})

    def test_multiple_splits_or_conflicting_benign_counts_are_rejected(self):
        for split, benign in [("test", 7), ("train", 8)]:
            with self.subTest(split=split, benign=benign):
                rows = self.windows([(1, "train", "A", 4, 7), (1, split, "B", 7, benign)])
                with self.assertRaisesRegex(ValueError, "conflicting"):
                    assign_benign(rows, ["A", "B"])

    def test_duplicate_class_window_is_rejected(self):
        rows = self.windows([(1, "train", "A", 4, 7)] * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assign_benign(rows, ["A"])

    def test_thresholds_and_benign_counts(self):
        rows = self.windows([
            (1, "train", "A", 4, 7), (2, "train", "A", 5, 3),
            (3, "train", "A", 10, 0), (4, "train", "A", 20, 8),
        ])
        stats = summarize(assign_benign(rows, ["A"]), ["A"]).set_index("split")
        values = stats.loc["train"]
        self.assertEqual(tuple(values[["usable_windows", "attack_ge5", "attack_ge10", "attack_ge20"]]),
                         (4, 3, 2, 1))
        self.assertEqual(values["median_attack"], 7.5)
        self.assertEqual((values["benign_windows"], values["benign_flows"]), (3, 18))
        self.assertEqual(stats.loc["test", "usable_windows"], 0)
        self.assertTrue(math.isnan(stats.loc["test", "median_attack"]))

    def test_frozen_permutations_and_pair_labels(self):
        protocol = read_json(STEP / "step1_protocol.json")
        seeds = protocol["random_order"]["seeds"]
        for name, config in protocol["datasets"].items():
            with self.subTest(dataset=name):
                eligible = config["eligible_classes"]
                frozen = config["random_orders"]
                self.assertEqual(list(frozen), list(map(str, seeds)))
                self.assertEqual(len({tuple(order) for order in frozen.values()}), 5)
                self.assertEqual(config["possible_full_orders"], math.factorial(len(eligible)))
                for seed in seeds:
                    self.assertEqual(frozen[str(seed)], seeded_order(eligible, seed))
                for pair in config["pairs"]:
                    self.assertTrue(pair["reverse"])
                    self.assertEqual(len(set(pair["classes"])), 2)
                    self.assertLessEqual(set(pair["classes"]), set(eligible))

    def test_relation_blocks_use_each_class_train_cut(self):
        columns = ["Attack", TIME, acs.HOST_SRC, acs.HOST_DST, *RELATION_FIELDS]
        observations = pd.DataFrame([
            ("x", 1, "s1", "d1", 20, 50, 6, 1),
            ("x", 2, "s2", "d2", 30, 50, 6, 2),
            ("y", 2, "s3", "d3", 20, 70, 6, 2),
            ("y", 3, "s4", "d4", 50, 50, 17, 1),
        ], columns=columns)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "flows.csv"
            observations.to_csv(path, index=False)
            observed = recompute_ton_relation_pair(
                path, {"x": {"t70": 2}, "y": {"t70": 3}}, ["x", "y"], {"x": 1, "y": 1})
        self.assertEqual(list(observed), acs.RELATION_BLOCKS)
        self.assertEqual(list(observed.values()), [0, 0, 0, 0, 1, 0, 1])
