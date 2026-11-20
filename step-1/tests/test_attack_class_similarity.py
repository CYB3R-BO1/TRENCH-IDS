#!/usr/bin/env python3
"""Small synthetic tests for attack_class_similarity.py (no dataset scan)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import attack_class_similarity as acs  # noqa: E402
import feature_schema as fs  # noqa: E402
import step1_3_uncertainty as u13  # noqa: E402
import ton_benign_audit as tba  # noqa: E402
import ton_class_contiguous as tcc  # noqa: E402
import ton_window_sweep as tws  # noqa: E402
import window_sweep as ws  # noqa: E402

TOY_ATTACKS = ["Alpha", "Beta", "Gamma"]
TOY_ALL = [acs.BENIGN] + TOY_ATTACKS


class TestReservoir(unittest.TestCase):
    def test_keeps_all_when_below_cap(self):
        rng = np.random.default_rng(0)
        res = np.zeros((10, 1), dtype=np.float32)
        rows = np.arange(6, dtype=np.float32).reshape(-1, 1)
        seen = acs._reservoir_update(res, 0, rows, rng, 10)
        self.assertEqual(seen, 6)
        np.testing.assert_array_equal(res[:6].ravel(), np.arange(6, dtype=np.float32))

    def test_size_and_membership_when_over_cap(self):
        rng = np.random.default_rng(42)
        cap, n = 4, 50
        res = np.full((cap, 1), -1.0, dtype=np.float32)
        seen = 0
        for i in range(n):
            seen = acs._reservoir_update(res, seen, np.array([[float(i)]], np.float32), rng, cap)
        self.assertEqual(seen, n)
        self.assertEqual(res.ravel().size, cap)

    def test_deterministic_for_same_seed(self):
        def run(seed):
            rng = np.random.default_rng(seed)
            res = np.zeros((5, 1), dtype=np.float32)
            seen = 0
            for i in range(20):
                seen = acs._reservoir_update(res, seen, np.array([[float(i)]], np.float32), rng, 5)
            return res.copy()
        np.testing.assert_array_equal(run(7), run(7))


class TestCategoricalValidation(unittest.TestCase):
    def test_valid_integral_in_range(self):
        valid, codes, kinds = acs.categorical_value_report("PROTOCOL", np.array([1.0, 6.0, 17.0]))
        self.assertTrue(valid.all())
        self.assertEqual(codes.tolist(), [1, 6, 17])
        self.assertEqual(int(kinds.sum()), 0)

    def test_fractional_is_invalid_not_rounded(self):
        valid, codes, kinds = acs.categorical_value_report("L4_DST_PORT", np.array([80.0, 80.4, 443.0]))
        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertEqual(int(codes[1]), 0)
        self.assertEqual(int(np.bincount(kinds, minlength=5)[3]), 1)

    def test_out_of_range_is_invalid(self):
        valid, _, kinds = acs.categorical_value_report("L4_SRC_PORT", np.array([0.0, 65535.0, 65536.0, -1.0]))
        self.assertEqual(valid.tolist(), [True, True, False, False])
        self.assertEqual(int(np.bincount(kinds, minlength=5)[4]), 2)

    def test_tcp_flags_reuses_flag_validator(self):
        valid, codes, kinds = acs.categorical_field_codes(
            "TCP_FLAGS", np.array([1.0, 2.0, 300.0, 1.7, np.nan]))
        self.assertEqual(valid.tolist(), [True, True, False, False, False])
        self.assertEqual(codes[valid].tolist(), [1, 2])
        self.assertIsNone(kinds)


class TestProbVector(unittest.TestCase):
    def test_sums_to_one_and_other_bucket(self):
        v = acs._prob_vector(Counter({1: 3, 2: 1}), [1, 2])
        self.assertAlmostEqual(v.sum(), 1.0)
        np.testing.assert_allclose(v, [0.75, 0.25, 0.0])

    def test_out_of_universe_goes_to_other(self):
        v = acs._prob_vector(Counter({1: 1, 99: 3}), [1])
        np.testing.assert_allclose(v, [0.25, 0.75])

    def test_empty_counter(self):
        v = acs._prob_vector(Counter(), [1, 2, 3])
        self.assertEqual(v.shape[0], 4)
        self.assertEqual(v.sum(), 0.0)


class TestRelationProfileConstruction(unittest.TestCase):
    def test_identity_free_same_multiset(self):
        a = Counter({"10.0.0.1": 3, "10.0.0.2": 1})
        b = Counter({"192.168.9.7": 1, "192.168.9.9": 3})
        va = acs.sorted_share_distribution(a, 4)
        vb = acs.sorted_share_distribution(b, 4)
        np.testing.assert_allclose(va, vb)
        np.testing.assert_allclose(va, [0.75, 0.25, 0.0, 0.0])

    def test_shape_matters(self):
        self.assertFalse(np.allclose(
            acs.sorted_share_distribution(Counter({"x": 2, "y": 2}), 4),
            acs.sorted_share_distribution(Counter({"x": 3, "y": 1}), 4)))

    def test_padding_and_empty(self):
        np.testing.assert_allclose(acs.sorted_share_distribution(Counter({"a": 1}), 5),
                                   [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(acs.sorted_share_distribution(Counter(), 3).tolist(), [0.0, 0.0, 0.0])


class TestEpisodeIntervals(unittest.TestCase):
    def test_two_episodes_merged_with_margin(self):
        self.assertEqual(acs.episode_intervals(np.array([0, 10, 20, 100_000, 100_010]), 1_000, 100),
                         [[-100, 120], [99_900, 100_110]])

    def test_single_episode(self):
        self.assertEqual(acs.episode_intervals(np.array([0, 1, 2, 3]), 1_000, 5), [[-5, 8]])

    def test_empty(self):
        self.assertEqual(acs.episode_intervals(np.array([], dtype=np.int64), 1_000, 5), [])

    def test_clip(self):
        self.assertEqual(acs.episode_intervals(np.array([10, 20]), 1_000, 100,
                                               clip_lo=5, clip_hi=25), [[5, 25]])


class TestBenignIntervalCounting(unittest.TestCase):
    def test_counts_benign_in_merged_intervals(self):
        benign = np.array([0, 5, 10, 15, 20, 100, 105], dtype=np.int64)
        iv = acs.episode_intervals(np.array([0, 20, 100, 105]), gap_ms=50, margin_ms=2)
        self.assertEqual(iv, [[-2, 22], [98, 107]])
        self.assertEqual(acs._count_in_intervals(benign, iv), 7)

    def test_disjoint_splits_do_not_share_benign(self):
        benign = np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.int64)
        a = acs.episode_intervals(np.array([0, 10]), 1_000, 100, clip_lo=0, clip_hi=10)
        b = acs.episode_intervals(np.array([20, 30]), 1_000, 100, clip_lo=20, clip_hi=30)
        self.assertEqual(acs._count_in_intervals(benign, a), 3)
        self.assertEqual(acs._count_in_intervals(benign, b), 3)


class TestContinuousInvalidMasks(unittest.TestCase):
    def test_mutually_exclusive(self):
        x = np.array([[np.nan, np.inf, -np.inf, -5.0, 3.0]])
        nan, inf, neg = acs.continuous_invalid_masks(x)
        self.assertEqual(int(nan.sum()), 1)
        self.assertEqual(int(inf.sum()), 2)
        self.assertEqual(int(neg.sum()), 1)
        self.assertFalse(bool((nan & inf).any() or (nan & neg).any() or (inf & neg).any()))


class TestCosineZeroVector(unittest.TestCase):
    def test_zero_vector_policy(self):
        sim = acs.cosine_matrix(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(sim[0, 0], 1.0)
        self.assertAlmostEqual(sim[0, 1], 0.0)
        self.assertAlmostEqual(sim[1, 1], 1.0)
        self.assertAlmostEqual(sim[1, 2], 0.0)
        self.assertTrue(np.isfinite(sim).all())


class TestDefinedCosine(unittest.TestCase):
    def test_uses_jointly_valid_features(self):
        z = np.array([[1.0, np.nan, 2.0], [1.0, np.nan, 2.0]])
        mat, zeros = acs.defined_cosine(z, ["A", "B"])
        self.assertAlmostEqual(mat[0, 1], 1.0)
        self.assertEqual(zeros, [])

    def test_no_joint_feature_raises(self):
        z = np.array([[1.0, np.nan], [np.nan, 1.0]])
        with self.assertRaises(ValueError):
            acs.defined_cosine(z, ["A", "B"])


class TestJSCoverage(unittest.TestCase):
    def test_one_missing_field_reduces_coverage(self):
        items = [
            ("a", [np.array([1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])], [True, True, True]),
            ("b", [np.array([0.5, 0.5]), np.zeros(2), np.array([0.5, 0.5])], [True, False, True]),
        ]
        agg, per, den = acs.aggregate_pairwise_js(items, 3)
        self.assertEqual(int(den[0, 1]), 1)
        self.assertEqual(int(den[0, 2]), 2)
        self.assertTrue(np.isfinite(agg).all())
        self.assertTrue(np.isnan(per["b"][0, 1]))


class TestFlagValidity(unittest.TestCase):
    def test_valid_and_invalid_classification(self):
        v = np.array([0, 6, 255, np.nan, np.inf, -1, 3.5, 256, 300])
        valid, codes, kinds = acs.flag_value_report(v)
        self.assertEqual(int(valid.sum()), 3)
        np.testing.assert_array_equal(codes[valid], [0, 6, 255])
        cnt = np.bincount(kinds, minlength=len(acs.FLAG_KIND_NAMES))
        self.assertEqual(list(cnt[:6]), [3, 1, 1, 1, 1, 2])

    def test_prevalence_uses_valid_denominator(self):
        n_mask, n_bits = len(acs.FLAG_MASKS), len(acs.FLAG_BITS)
        flagbit = np.zeros((1, n_mask * n_bits), dtype=np.int64)
        flagbit[0, 0] = 1
        prev, den = acs.flag_bit_prevalence(flagbit, np.array([[2, 4, 0]]))
        self.assertAlmostEqual(prev[0, 0], 0.5)
        self.assertEqual(den[0, n_bits], 4)
        self.assertAlmostEqual(prev[0, 2 * n_bits], 0.0)


class TestFlagSupportGaps(unittest.TestCase):
    def test_reports_zero_valid_combinations(self):
        valid = np.array([[5, 0, 2], [0, 0, 0]], dtype=np.int64)
        self.assertEqual(acs.flag_support_gaps(valid, ["A", "B"], ["m0", "m1", "m2"]),
                         [("A", "m1"), ("B", "m0"), ("B", "m1"), ("B", "m2")])


class TestLabelValidation(unittest.TestCase):
    def test_clean(self):
        counts, bad, mismatch = acs.validate_labels(np.array([0, 1, 0, 1]),
                                                    np.array(["Benign", "dos", "Benign", "xss"]))
        self.assertEqual((bad, mismatch, counts[1]), (0, 0, 2))

    def test_unexpected_domain(self):
        _, bad, _ = acs.validate_labels(np.array([0, 1, 2]), np.array(["Benign", "dos", "xss"]))
        self.assertEqual(bad, 1)

    def test_consistency_mismatch(self):
        _, bad, mismatch = acs.validate_labels(np.array([1, 0]), np.array(["Benign", "dos"]))
        self.assertEqual((bad, mismatch), (0, 2))


class TestTimestampValidation(unittest.TestCase):
    def test_valid_integers(self):
        t = acs.validate_timestamp_column(np.array([0.0, 1000.0, 2.0 ** 40]), "T")
        self.assertEqual(t.dtype, np.int64)
        self.assertEqual(t.tolist(), [0, 1000, 2 ** 40])

    def test_fractional_rejected(self):
        with self.assertRaises(ValueError):
            acs.validate_timestamp_column(np.array([1.0, 1.5]), "T")

    def test_non_finite_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                acs.validate_timestamp_column(np.array([1.0, bad]), "T")

    def test_out_of_float64_exact_rejected(self):
        with self.assertRaises(ValueError):
            acs.validate_timestamp_column(np.array([2.0 ** 60]), "T")

    def test_order_rejected(self):
        with self.assertRaises(ValueError):
            acs.validate_timestamp_order(np.array([10, 20]), np.array([10, 19]))
        acs.validate_timestamp_order(np.array([10, 20]), np.array([10, 20]))


class TestIdentityValidation(unittest.TestCase):
    def test_null_empty_and_blank(self):
        m = acs.invalid_identity_mask(np.array(["1.2.3.4", "", "   ", None, np.nan], dtype=object))
        self.assertEqual(m.tolist(), [False, True, True, True, True])


class TestClassDiscovery(unittest.TestCase):
    def test_discovers_sorted_non_benign_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "toy.csv"
            pd.DataFrame({"Attack": ["Benign", "Zeta", "Alpha", "Benign", "Zeta"]}).to_csv(p, index=False)
            self.assertEqual(acs.discover_attack_classes(p, {"chunk": 2}), ["Alpha", "Zeta"])

    def test_only_benign_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "toy.csv"
            pd.DataFrame({"Attack": ["Benign", "Benign"]}).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                acs.discover_attack_classes(p, {"chunk": 2})


class TestFeatureSchema(unittest.TestCase):
    def test_partition(self):
        self.assertEqual(len(fs.MAGNITUDE_FEATURES), 37)
        for f in fs.IDENTIFIER_FEATURES + fs.BITMASK_FEATURES + fs.CATEGORICAL_NUMERIC:
            self.assertNotIn(f, fs.MAGNITUDE_FEATURES)
        self.assertEqual(fs.IDENTIFIER_FEATURES, ["DNS_QUERY_ID"])
        for f in ("ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE"):
            self.assertIn(f, fs.CATEGORICAL_FEATURES)
            self.assertIn(f, acs.CATEGORICAL_RANGES)
        self.assertIn("TCP_FLAGS", fs.BITMASK_FEATURES)


class TestGlobalSplit(unittest.TestCase):
    def _cfg(self):
        return {"split": (0.5, 0.8), "chunk": 3, "max_rows": None}

    def test_all_flow_quantiles_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "toy.csv"
            pd.DataFrame({"FLOW_START_MILLISECONDS": list(range(100))}).to_csv(p, index=False)
            t70, t85 = acs.compute_global_split_cuts(p, self._cfg(), 10)
            self.assertEqual((t70 % 10, t85 % 10), (0, 0))  # aligned
            self.assertLess(t70, t85)

    def test_fractional_timestamp_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "toy.csv"
            pd.DataFrame({"FLOW_START_MILLISECONDS": [0, 1.5]}).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                acs.compute_global_split_cuts(p, self._cfg(), 1)


class TestWindowFeasibility(unittest.TestCase):
    def test_total_vs_class_and_split_isolation(self):
        # cuts aligned: train < 10, val < 20; window 10
        ts_parts = [
            [np.array([5], dtype=np.int64)],                 # Benign
            [np.array([5, 15], dtype=np.int64)],             # Alpha (train w0, val w1)
            [np.array([15], dtype=np.int64)],                # Beta (val w1 only)
        ]
        df = acs.winmod.window_feasibility(ts_parts, ["Benign", "Alpha", "Beta"], (10, 20), 10, 2)
        a = df[(df["class"] == "Alpha")]
        train = a[a["split"] == "train"].iloc[0]
        val = a[a["split"] == "val"].iloc[0]
        # Beta is only in val -> Alpha's train window is not mixed
        self.assertEqual(train["mixed_windows"], 0)
        self.assertEqual(val["mixed_windows"], 1)
        # class count 1 < 2 but total flows in val window = 2 -> graph-sized
        self.assertEqual(val["class_windows_ge_min"], 0)
        self.assertEqual(val["graph_windows_ge_min_total_flows"], 1)


class TestWassersteinScaling(unittest.TestCase):
    def test_standardised_aggregation(self):
        d = _toy_state()
        cfg = _toy_cfg()
        mats = acs.build_matrices(d, cfg)
        feat = acs.DIST_FEATURES[0]
        p = d["magnitude_features"].index(feat)
        gv = d["g_valid"]
        gmean = np.divide(d["g_sum"], gv, out=np.zeros(d["F"]), where=gv > 0)
        gvar = np.divide(d["g_sumsq"], gv, out=np.zeros(d["F"]), where=gv > 0) - gmean ** 2
        gstd = np.sqrt(np.clip(gvar, 0, None))
        gstd[gstd == 0] = 1.0
        ci_a, ci_b = 1, 2  # TOY_ATTACKS[0], [1]
        sa = d["res"][ci_a][:, p][np.isfinite(d["res"][ci_a][:, p])]
        sb = d["res"][ci_b][:, p][np.isfinite(d["res"][ci_b][:, p])]
        expected = wasserstein_distance((sa - gmean[p]) / gstd[p], (sb - gmean[p]) / gstd[p])
        self.assertAlmostEqual(mats["wasserstein_per_feature"][feat][0, 1], expected)
        self.assertEqual(mats["wasserstein_pair_coverage"].shape, (len(TOY_ATTACKS), len(TOY_ATTACKS)))


class TestTrainSupportScope(unittest.TestCase):
    def test_zero_support_class_is_excluded_but_reported(self):
        d = _toy_state()
        d["ccount_split"][1, 0] = 0  # Alpha has no training flows
        supported, scope = acs.train_support_scope(d, _toy_cfg())
        self.assertNotIn(TOY_ATTACKS[0], supported)
        status = dict(zip(scope["class"], scope["status"]))
        self.assertEqual(status[TOY_ATTACKS[0]], "zero_train_support")
        undef = acs.undefined_pairs(scope, supported)
        self.assertTrue((undef["class_b"] == TOY_ATTACKS[0]).any())

    def test_low_support_class_is_excluded_with_reason(self):
        d = _toy_state()
        cfg = _toy_cfg()
        cfg["split_floor"] = 5  # toy train counts are 3 -> low
        supported, scope = acs.train_support_scope(d, cfg)
        self.assertEqual(supported, [])
        status = dict(zip(scope["class"], scope["status"]))
        self.assertEqual(status[TOY_ATTACKS[0]], "low_train_support")


class TestCosineCoverage(unittest.TestCase):
    def test_consistent_feature_set(self):
        d = _toy_state()
        d["c_valid"][1, 0] = 0  # feature 0 invalid for one class
        mats = acs.build_matrices(d, _toy_cfg())
        used = [d["magnitude_features"][p] for p in mats["cosine_consistent_features"]]
        self.assertNotIn(d["magnitude_features"][0], used)
        self.assertEqual(mats["cosine_pair_coverage"].shape, (len(TOY_ATTACKS), len(TOY_ATTACKS)))


class TestCategoricalProtocolCodes(unittest.TestCase):
    def test_protocol_codes_are_categorical(self):
        self.assertIn("ICMP_TYPE", fs.CATEGORICAL_FEATURES)
        self.assertIn("DNS_QUERY_TYPE", fs.CATEGORICAL_FEATURES)
        self.assertNotIn("ICMP_TYPE", fs.MAGNITUDE_FEATURES)
        self.assertNotIn("DNS_QUERY_TYPE", fs.MAGNITUDE_FEATURES)


class TestMaxRows(unittest.TestCase):
    def test_iter_csv_respects_max_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "toy.csv"
            pd.DataFrame({"x": list(range(10))}).to_csv(p, index=False)
            rows = sum(len(c) for c in acs._iter_csv(p, {"chunk": 3, "max_rows": 5}))
            self.assertEqual(rows, 5)


def _toy_state(cap: int = 8) -> dict:
    K = len(TOY_ALL)
    F = len(acs.MAGNITUDE_FEATURES)
    rng = np.random.default_rng(0)
    ts = np.array([0, 10, 20, 40, 70, 90], dtype=np.int64)
    ccount_split = np.zeros((K, 3), dtype=np.int64)
    ccount_split[:, 0], ccount_split[:, 1], ccount_split[:, 2] = 3, 1, 2
    return {
        "n_rows": 100, "dataset_name": "toy",
        "all_classes": TOY_ALL, "matrix_classes": list(TOY_ATTACKS),
        "magnitude_features": list(acs.MAGNITUDE_FEATURES), "F": F, "K": K, "cuts": (30, 80),
        "g_valid": np.full(F, 100, dtype=np.int64), "g_sum": np.zeros(F), "g_sumsq": np.full(F, 100.0),
        "c_valid": np.full((K, F), 10, dtype=np.int64),
        "c_sum": rng.normal(size=(K, F)) * 10.0,
        "c_sumsq": np.full((K, F), 500.0),
        "res": rng.normal(size=(K, cap, F)).astype(np.float32),
        "seen": np.full(K, cap, dtype=np.int64),
        "have_res": np.array([n != acs.BENIGN for n in TOY_ALL]),
        "g_nan": np.zeros(F, dtype=np.int64), "g_inf": np.zeros(F, dtype=np.int64),
        "g_neg": np.zeros(F, dtype=np.int64),
        "c_nan": np.zeros((K, F), dtype=np.int64), "c_inf": np.zeros((K, F), dtype=np.int64),
        "c_neg": np.zeros((K, F), dtype=np.int64),
        "g_cat_kind": np.zeros((len(acs.CATEGORICAL_FIELDS), len(acs.CATEGORICAL_KIND_NAMES)), dtype=np.int64),
        "c_cat_kind": np.zeros((K, len(acs.CATEGORICAL_FIELDS), len(acs.CATEGORICAL_KIND_NAMES)), dtype=np.int64),
        "g_flag_kind": np.zeros((len(acs.FLAG_MASKS), len(acs.FLAG_KIND_NAMES)), dtype=np.int64),
        "c_flag_kind": np.tile(np.array([10, 0, 0, 0, 0, 0], dtype=np.int64),
                               (K, len(acs.FLAG_MASKS), 1)),
        "src_ct": [Counter({f"10.0.0.{i}": i + 1 for i in range(4)}) for _ in range(K)],
        "dst_ct": [Counter({f"10.0.1.{i}": i + 1 for i in range(5)}) for _ in range(K)],
        "pair_ct": [Counter({(f"10.0.0.{i}", f"10.0.1.{i}"): i + 1 for i in range(4)}) for _ in range(K)],
        "cat_ct": [[Counter({1: 5, 2: 3, 6: 2}) for _ in acs.CATEGORICAL_FIELDS] for _ in range(K)],
        "flagbit": np.ones((K, len(acs.FLAG_BIT_NAMES)), dtype=np.int64),
        "c_flag_valid": np.full((K, len(acs.FLAG_MASKS)), 10, dtype=np.int64),
        "ts_parts": [[ts.copy()] for _ in range(K)],
        "ccount_split": ccount_split,
        "cat_valid_split": np.full((K, 3, len(acs.CATEGORICAL_FIELDS)), 5, dtype=np.int64),
        "flag_valid_split": np.full((K, 3, len(acs.FLAG_MASKS)), 5, dtype=np.int64),
        "src_ct_split": [[Counter({"10.0.0.0": 1}) for _ in range(3)] for _ in range(K)],
        "dst_ct_split": [[Counter({"10.0.1.0": 1}) for _ in range(3)] for _ in range(K)],
        "label_counts": Counter({0: 70, 1: 30}),
        "attack_counts": Counter({acs.BENIGN: 70, **{a: 10 for a in TOY_ATTACKS}}),
        "label_mismatch": 0,
    }


def _toy_cfg(cap: int = 8) -> dict:
    return {"reservoir_cap": cap, "window_ms": 10, "window_sweep_ms": [10, 20],
            "split": (0.5, 0.8), "split_floor": 1, "min_feature_samples": 2,
            "min_train_flows": 1, "cat_topk": 0,
            "margin_min": 15.0, "episode_gap_min": 15.0,
            "sensitivity_gaps": [1, 15], "sensitivity_margins": [0, 15],
            "min_graph_flows": 1, "seed": 0, "chunk": 1_000, "out": "x",
            "dataset": "toy", "dataset_name": "toy", "taxonomy_csv": None, "max_rows": None}


class TestPipelineSmoke(unittest.TestCase):
    def setUp(self):
        self.d = _toy_state()
        self.cfg = _toy_cfg()
        self.M = len(TOY_ATTACKS)

    def test_matrices_shapes_and_symmetry(self):
        mats = acs.build_matrices(self.d, self.cfg)
        for key in ("cosine", "wasserstein", "jensen_shannon", "relation_profile", "flag_bit_similarity"):
            m = mats[key]
            self.assertEqual(m.shape, (self.M, self.M))
            np.testing.assert_allclose(m, m.T, atol=1e-9)
        for key in ("cosine", "relation_profile", "flag_bit_similarity"):
            np.testing.assert_allclose(np.diag(mats[key]), 1.0)
        self.assertGreaterEqual(mats["relation_profile"].min(), 0.0)
        self.assertLessEqual(mats["relation_profile"].max(), 1.0)
        self.assertEqual(mats["flag_support_gaps"], [])

    def test_relation_blocks_directional(self):
        mats = acs.build_matrices(self.d, self.cfg)
        self.assertEqual(len(acs.RELATION_BLOCKS), 7)
        self.assertIn("flow_to_src_port", mats["relation_block_distance"])
        self.assertIn("source_host_degree", mats["relation_block_distance"])
        self.assertIn("dest_host_degree", mats["relation_block_distance"])

    def test_feasibility_uses_global_split(self):
        feas, sweep = acs.build_feasibility(self.d, self.cfg)
        for col in ("train_flows", "val_flows", "test_flows", "min_split_flows",
                    "attack_floor_ok", "benign_ok_train", "benign_ok_val", "benign_ok_test"):
            self.assertIn(col, feas.columns)
        self.assertEqual(len(sweep), len(TOY_ALL) * len(self.cfg["window_sweep_ms"]))

    def test_margin_gap_sensitivity(self):
        sens = acs.build_margin_gap_sensitivity(self.d, self.cfg)
        n = len(self.cfg["sensitivity_gaps"]) * len(self.cfg["sensitivity_margins"])
        self.assertEqual(len(sens), len(TOY_ALL) * n)
        for col in ("benign_contemp_train", "benign_contemp_val", "benign_contemp_test"):
            self.assertIn(col, sens.columns)

    def test_output_writer_emits_all_files(self):
        d, cfg = self.d, self.cfg
        mats = acs.build_matrices(d, cfg)
        feas, sweep = acs.build_feasibility(d, cfg)
        sens = acs.build_margin_gap_sensitivity(d, cfg)
        window_feas = acs.winmod.window_feasibility(d["ts_parts"], d["all_classes"], d["cuts"],
                                                    cfg["window_ms"], cfg["min_graph_flows"])
        prov = {"data_path": "toy", "data_sha256": "0", "data_size_mb": 0.0}
        with tempfile.TemporaryDirectory() as tmp:
            old_out, old_fig = acs.OUT_DIR, acs.FIG_DIR
            acs.OUT_DIR, acs.FIG_DIR = Path(tmp), Path(tmp) / "figures"
            try:
                acs.write_outputs(d, mats, feas, sweep, sens, window_feas, cfg, prov)
            finally:
                acs.OUT_DIR, acs.FIG_DIR = old_out, old_fig
            expected = [
                "standardized_mean_profile_cosine.csv", "wasserstein_distance.csv",
                "wasserstein_per_feature.csv", "wasserstein_feature_support.csv",
                "wasserstein_pair_coverage.csv",
                "cosine_feature_support.csv", "cosine_pair_coverage.csv",
                "cosine_consistent_features.json",
                "jensen_shannon_distance.csv", "jensen_shannon_per_field.csv",
                "relation_profile_similarity.csv", "relation_block_distance.csv",
                "host_identity_overlap.csv", "host_identity_overlap_source.csv",
                "tcp_flag_bit_prevalence.csv", "tcp_flag_valid_counts.csv", "flag_support_gaps.csv",
                "invalid_categorical_values.csv", "invalid_flag_values.csv", "feature_validity.csv",
                "js_defined_field_count.csv", "relation_defined_block_count.csv",
                "categorical_compression.csv", "provenance.json", "feature_schema.json",
                "split_manifest.json", "class_scope.csv", "undefined_pairs.csv",
                "sample_feasibility.csv", "window_occupancy.csv",
                "window_feasibility.csv", "margin_gap_sensitivity.csv", "pair_comparison.csv",
                "rankings_cosine.csv", "rankings_wasserstein.csv",
                "rankings_js.csv", "rankings_relation.csv", "rankings_flag_bits.csv",
                "similarity_results.json", "categorical_universe.json",
            ]
            for name in expected:
                self.assertTrue((Path(tmp) / name).exists(), f"missing {name}")


class TestConfigValidation(unittest.TestCase):
    def _cfg(self, **over):
        base = _toy_cfg()
        base.update(over)
        return base

    def test_ok(self):
        acs.validate_cfg(_toy_cfg())

    def test_bad_split_order(self):
        with self.assertRaises(SystemExit):
            acs.validate_cfg(self._cfg(split=(0.90, 0.50)))

    def test_non_finite_margin(self):
        with self.assertRaises(SystemExit):
            acs.validate_cfg(self._cfg(margin_min=float("nan")))

    def test_negative_seed(self):
        with self.assertRaises(SystemExit):
            acs.validate_cfg(self._cfg(seed=-1))


class TestExcludeFeatures(unittest.TestCase):
    def _write(self, tmp):
        p = Path(tmp) / "toy.csv"
        pd.DataFrame({"A": [1]}).to_csv(p, index=False)
        return p

    def test_unknown_feature_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _toy_cfg()
            cfg["exclude_features"] = ["NOPE"]
            with self.assertRaises(ValueError) as cm:
                acs.extract(self._write(tmp), cfg, ["Alpha"], (0, 1))
            self.assertIn("unknown --exclude-features", str(cm.exception))

    def test_known_feature_passes_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _toy_cfg()
            cfg["exclude_features"] = ["SRC_TO_DST_SECOND_BYTES"]
            with self.assertRaises(ValueError) as cm:
                acs.extract(self._write(tmp), cfg, ["Alpha"], (0, 1))
            self.assertIn("missing from CSV", str(cm.exception))


class TestWindowSweep(unittest.TestCase):
    def test_window_counts(self):
        ts = np.array([0, 1, 2, 1000, 1001, 2500], dtype=np.int64)
        ccode = np.array([0, 1, 0, 1, 1, 0], dtype=np.int16)
        total, benign, counts = ws.window_counts(ts, ccode, 1, 1000)
        np.testing.assert_array_equal(total, [3, 2, 1])
        np.testing.assert_array_equal(benign, [2, 0, 1])
        np.testing.assert_array_equal(counts[1], [1, 2, 0])

    def test_split_of_windows_alignment(self):
        split = ws.split_of_windows(10, 1000, 3000, 6000)
        np.testing.assert_array_equal(split, [0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        with self.assertRaises(ValueError):
            ws.split_of_windows(10, 4000, 3000, 6000)


class TestStep13(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(u13._entropy(Counter({1: 5})), 0.0)
        self.assertEqual(u13._entropy(Counter()), 0.0)
        self.assertAlmostEqual(u13._entropy(Counter({1: 5, 2: 5})), 1.0)

    def test_groups_partition(self):
        keys = np.array([5, 5, 3, 3, 3, 9])
        got = {k: list(rows) for k, rows in u13._groups(keys, np.arange(6))}
        self.assertEqual(got[3], [2, 3, 4])
        self.assertEqual(got[5], [0, 1])
        self.assertEqual(got[9], [5])

    def test_window_counts_base_offset(self):
        total, benign, counts = ws.window_counts(
            np.array([1000, 1001, 3000], dtype=np.int64),
            np.array([0, 1, 0], dtype=np.int16), 1, 1000, base=1000)
        # base=1000, w=1000 -> window ids (t-1000)//1000 = [0, 0, 2]; id 1 is empty
        np.testing.assert_array_equal(total, [2, 0, 1])
        np.testing.assert_array_equal(benign, [1, 0, 1])
        np.testing.assert_array_equal(counts[1], [1, 0, 0])


    def test_rank_stats(self):
        arr = np.zeros((2, 3, 3))
        arr[0, 0, 1], arr[0, 0, 2], arr[0, 1, 2] = 0.9, 0.5, 0.1
        arr[1, 0, 1], arr[1, 0, 2], arr[1, 1, 2] = 0.2, 0.8, 0.1
        pairs = [("A", "B", 0, 1), ("A", "C", 0, 2), ("B", "C", 1, 2)]
        _, ranks = u13._rank_stats(arr, pairs, sim=True)
        self.assertEqual(list(ranks[0]), [1, 2, 3])
        self.assertEqual(list(ranks[1]), [2, 1, 3])


class TestTonContiguous(unittest.TestCase):
    def test_snap_to_60s_grid(self):
        self.assertEqual(tcc._snap(1556029139999), 1556029080000)
        self.assertEqual(tcc._snap(1556029140000), 1556029140000)
        self.assertEqual(tcc._snap(1556029140000) % tcc.ALIGN_MS, 0)


class TestTonBenignAudit(unittest.TestCase):
    def test_split_of_window(self):
        self.assertEqual(tba.split_of(0, 60000, 120000), 0)
        self.assertEqual(tba.split_of(1, 60000, 120000), 1)
        self.assertEqual(tba.split_of(2, 60000, 120000), 2)


class TestTonWindowSweep(unittest.TestCase):
    def test_analyse_per_class_split(self):
        total = np.array([5, 3, 0, 4]); benign = np.array([3, 1, 0, 2])
        cls = np.array([[2, 2, 0, 0], [0, 0, 0, 2]])
        cuts = {"A": (2000, 3000), "B": (2000, 3000)}
        floors = {"A": True, "B": False}
        comp, nat, acls, filt, ratio, mix = tws.analyse(
            1000, 0, total, benign, cls, ["A", "B"], cuts, floors, [2, 5, 10, 20])
        a_train = [r for r in nat if r["class"] == "A" and r["split"] == "train"][0]
        self.assertEqual(a_train["windows"], 2)
        self.assertEqual(a_train["attack_flows"], 4)
        b_test = [r for r in nat if r["class"] == "B" and r["split"] == "test"][0]
        self.assertEqual(b_test["attack_flows"], 2)
        self.assertEqual([r for r in acls if r["class"] == "A" and r["split"] == "train"][0]["windows_class_ge2"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
