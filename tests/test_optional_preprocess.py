"""Precomputed signals and independently selected transformations stay separate."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import RISK_FIELDS
from fundamental_factor.preprocess import prepare_signals
from tests.fixtures import signal_fixture


DIRECT = {"winsorize": False, "standardize": False}


class OptionalPreprocessTests(unittest.TestCase):
    def test_direct_signal_preserves_values_and_ties_and_filters_only_invalid_factors(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=10)
        frame["value"] = [-1e9, -5., 0., 0., 2., 5., 1e9, np.nan, np.inf, -np.inf]
        frame["industry"] = None
        frame[RISK_FIELDS] = np.nan
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["raw"], sample_policy="independent")
        self.assertEqual(len(signals), 7)
        np.testing.assert_array_equal(signals.signal, signals.value)
        self.assertEqual(signals.loc[signals.value.eq(0), "group"].nunique(), 1)
        self.assertEqual(set(diagnostics.winsor_method), {"disabled"})
        self.assertEqual(set(diagnostics.status), {"ok"})

    def test_winsorization_and_standardization_switches_are_independent(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=101)
        frame["value"] = np.r_[np.linspace(-2, 2, 100), 1000.]
        clipped, _ = prepare_signals(frame, {"winsorize": True, "standardize": False}, variants=["raw"])
        scaled, _ = prepare_signals(frame, {"winsorize": False, "standardize": True}, variants=["raw"])
        direct, _ = prepare_signals(frame, DIRECT, variants=["raw"])
        self.assertLess(clipped.signal.max(), 20)
        self.assertEqual(direct.signal.max(), 1000)
        np.testing.assert_array_equal(clipped.signal.iloc[:100], frame.value.iloc[:100])
        self.assertAlmostEqual(scaled.signal.mean(), 0)
        self.assertAlmostEqual(scaled.signal.std(ddof=0), 1)
        pd.testing.assert_series_equal(scaled.signal.rank(), direct.signal.rank())

    def test_no_standardization_keeps_industry_residual_scale(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=8)
        frame["industry"] = ["A"] * 4 + ["B"] * 4
        frame["value"] = [-10., 0., 0., 10., 100., 100., 100., 104.]
        signals, _ = prepare_signals(frame, DIRECT, variants=["industry"], include_float_mcap=False,
                                     sample_policy="independent")
        np.testing.assert_allclose(signals.signal, [-10., 0., 0., 10., -1., -1., -1., 3.], atol=1e-12)
        self.assertGreater(signals.signal.std(ddof=0), 4)
        self.assertLess(np.abs(signals.groupby("industry").signal.mean()).max(), 1e-12)

    def test_exactly_explained_factor_does_not_create_roundoff_rank_signal(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=8)
        frame["industry"] = ["A"] * 4 + ["B"] * 4
        frame["value"] = [7.] * 4 + [11.] * 4
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["raw", "industry"],
                                               include_float_mcap=False, sample_policy="independent")
        self.assertTrue(signals.loc[signals.variant.eq("industry"), "signal"].eq(0).all())
        self.assertEqual(diagnostics.loc[diagnostics.variant.eq("industry"), "status"].iloc[0], "constant")
        np.testing.assert_array_equal(signals.loc[signals.variant.eq("raw"), "signal"], frame.value)
        # Small magnitude alone must not trigger neutral residual zeroing.
        frame["value"] = np.array([-10., 0., 0., 10., 100., 100., 100., 104.]) * 1e-14
        small, _ = prepare_signals(frame, DIRECT, variants=["industry"], include_float_mcap=False,
                                   sample_policy="independent")
        np.testing.assert_allclose(small.signal, np.array([-10., 0., 0., 10., -1., -1., -1., 3.]) * 1e-14,
                                   rtol=1e-12, atol=1e-26)

    def test_raw_results_do_not_depend_on_requested_control_variants(self):
        frame, _ = signal_fixture(weeks_count=2, stocks=80)
        frame.loc[frame.code.isin(["000000.SZ", "000001.SZ"]), "industry"] = None
        frame.loc[frame.code.eq("000002.SZ"), RISK_FIELDS[0]] = np.nan
        frame.loc[frame.code.eq("000003.SZ"), "float_mcap"] = np.nan
        frame.loc[frame.code.eq("000004.SZ"), "value"] = 1e8
        for settings in (DIRECT, {"winsorize": True, "standardize": True}):
            with self.subTest(settings=settings):
                alone, _ = prepare_signals(frame, settings, variants=["raw"], sample_policy="independent")
                combined, _ = prepare_signals(frame, settings, variants=["raw", "industry", "industry_size", "full"],
                                               sample_policy="independent")
                raw = combined[combined.variant.eq("raw")].reset_index(drop=True)
                pd.testing.assert_frame_equal(alone.reset_index(drop=True), raw, check_exact=True)
                sizes = combined.groupby(["entry", "variant"]).size().unstack()
                self.assertTrue(sizes.raw.eq(80).all())
                self.assertTrue(sizes.industry.eq(78).all())
                self.assertTrue(sizes.industry_size.eq(77).all())
                self.assertTrue(sizes.full.eq(76).all())

    def test_unidentifiable_neutralization_does_not_suppress_raw(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=8)
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["raw", "full"],
                                               sample_policy="independent")
        self.assertEqual(set(signals.variant), {"raw"})
        np.testing.assert_array_equal(signals.signal, signals.value)
        failed = diagnostics[diagnostics.variant.eq("full")].iloc[0]
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.reason, "insufficient_cross_section_degrees_of_freedom")
        common, common_diagnostics = prepare_signals(frame, DIRECT, variants=["raw", "full"],
                                                     sample_policy="common")
        self.assertTrue(common.empty)
        self.assertEqual(common_diagnostics.iloc[0].variant, "all")

    def test_missing_control_columns_skip_only_the_affected_variant(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=40)
        frame = frame.drop(columns=RISK_FIELDS)
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["raw", "industry", "full"],
                                               include_float_mcap=False, sample_policy="independent")
        self.assertEqual(set(signals.variant), {"raw", "industry"})
        failed = diagnostics[diagnostics.variant.eq("full")].iloc[0]
        self.assertEqual(failed.status, "failed")
        self.assertTrue(failed.reason.startswith("missing_control_columns:"))
        with self.assertRaisesRegex(ValueError, "missing columns"):
            prepare_signals(frame, DIRECT, variants=["raw", "full"], sample_policy="common")

    def test_no_control_observations_and_missing_factors_are_reported_without_zero_filling(self):
        frame, _ = signal_fixture(weeks_count=2, stocks=40)
        first = frame.entry.min()
        frame.loc[frame.entry.eq(first), "industry"] = " "
        frame.loc[frame.entry.ne(first), "value"] = np.nan
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["raw", "industry"],
                                               include_float_mcap=False, sample_policy="independent")
        self.assertEqual(len(signals), 40)
        self.assertTrue(signals.entry.eq(first).all())
        self.assertTrue(signals.variant.eq("raw").all())
        empty = diagnostics[diagnostics.status.eq("empty")]
        self.assertEqual(len(empty), 3)
        self.assertEqual(set(empty.reason), {"no_valid_factor_values", "no_valid_control_observations"})

    def test_legacy_defaults_equal_explicit_processing_and_common_policy(self):
        frame, _ = signal_fixture(weeks_count=2, stocks=80)
        original, old_diagnostics = prepare_signals(frame)
        explicit, diagnostics = prepare_signals(frame, {"winsorize": True, "standardize": True}, sample_policy="common")
        pd.testing.assert_frame_equal(original, explicit, check_exact=True)
        pd.testing.assert_frame_equal(old_diagnostics, diagnostics, check_exact=True)
        frame.loc[0, "value"] = np.nan
        with self.assertRaisesRegex(ValueError, "complete common sample"):
            prepare_signals(frame)

    def test_invalid_flags_and_sample_policy_fail_explicitly(self):
        frame, _ = signal_fixture(weeks_count=1)
        for settings in ({"winsorize": "false"}, {"standardize": 0}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "must be boolean"):
                prepare_signals(frame, settings)
        with self.assertRaisesRegex(ValueError, "sample_policy"):
            prepare_signals(frame, sample_policy="other")


if __name__ == "__main__":
    unittest.main()
