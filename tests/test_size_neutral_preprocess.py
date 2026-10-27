"""Market-cap neutrality uses log size without imposing industry dependencies."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import RISK_FIELDS
from fundamental_factor.preprocess import control_matrix, prepare_signals
from tests.fixtures import signal_fixture


DIRECT = {"winsorize": False, "standardize": False}


class SizeNeutralPreprocessTests(unittest.TestCase):
    def test_size_only_needs_no_industry_or_barra_for_either_sample_policy(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=24)
        frame = frame.drop(columns=["industry", *RISK_FIELDS])
        frame["value"] = 2 + 4 * np.log(frame.float_mcap) + np.sin(np.arange(len(frame)))
        for policy in ("common", "independent"):
            with self.subTest(policy=policy):
                signals, diagnostics = prepare_signals(frame, DIRECT, variants=["raw", "size"],
                                                       include_float_mcap=False, sample_policy=policy)
                self.assertEqual(set(signals.variant), {"raw", "size"})
                self.assertTrue(diagnostics.status.eq("ok").all())
                original = signals[signals.variant.eq("raw")]
                np.testing.assert_array_equal(original.signal, original.value)
                residual = signals[signals.variant.eq("size")]
                log_cap = np.log(residual.float_mcap.to_numpy())
                self.assertAlmostEqual(residual.signal.mean(), 0, places=11)
                self.assertAlmostEqual(np.dot(residual.signal, log_cap - log_cap.mean()), 0, places=10)

    def test_size_proxy_removes_log_size_exposure_and_keeps_residual_scale(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=40)
        frame = frame.drop(columns=["industry", "float_mcap", *RISK_FIELDS])
        log_cap = np.linspace(17, 25, len(frame))
        frame["market_cap_proxy"] = np.exp(log_cap)
        noise = np.cos(np.arange(len(frame))) * 2
        values = 7 + 3.5 * log_cap + noise
        frame["value"] = values
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["size"],
                                               sample_policy="independent", market_cap_column="market_cap_proxy")
        centered = log_cap - log_cap.mean()
        slope = np.dot(centered, values - values.mean()) / np.dot(centered, centered)
        expected = values - values.mean() - slope * centered
        np.testing.assert_allclose(signals.signal, expected, atol=5e-13)
        self.assertEqual(int(diagnostics.iloc[0]["columns"]), 2)
        controls, names = control_matrix(frame, include_industry=False, market_cap_column="market_cap_proxy")
        self.assertEqual(names, ["intercept", "log_market_cap_proxy"])
        np.testing.assert_allclose(controls.T @ signals.signal, 0, atol=1e-11)

    def test_industry_size_and_full_accept_proxy_without_actual_float_cap(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=100)
        frame = frame.rename(columns={"float_mcap": "market_cap_proxy"})
        frame["value"] += 3 * np.log(frame.market_cap_proxy) + frame.industry.map(
            {"银行": 1., "工业": 3., "消费": -2., "科技": 6.})
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["industry_size", "full"],
                                               sample_policy="independent", market_cap_column="market_cap_proxy")
        self.assertEqual(set(signals.variant), {"industry_size", "full"})
        self.assertTrue(diagnostics.status.eq("ok").all())
        for variant, group in signals.groupby("variant"):
            controls, names = control_matrix(group, full=variant == "full", market_cap_column="market_cap_proxy")
            self.assertIn("log_market_cap_proxy", names)
            self.assertNotIn("log_float_mcap", names)
            np.testing.assert_allclose(controls.T @ group.signal, 0, atol=1e-10)

    def test_invalid_proxy_drops_only_size_and_preserves_raw_exactly(self):
        frame, _ = signal_fixture(weeks_count=2, stocks=20)
        frame = frame.rename(columns={"float_mcap": "market_cap_proxy"})
        frame["industry"] = None
        frame[RISK_FIELDS] = np.nan
        bad = {"000000.SZ": np.nan, "000001.SZ": np.inf, "000002.SZ": 0., "000003.SZ": -10.}
        for code, value in bad.items():
            frame.loc[frame.code.eq(code), "market_cap_proxy"] = value
        baseline, _ = prepare_signals(frame, DIRECT, variants=["raw"], sample_policy="independent")
        signals, _ = prepare_signals(frame, DIRECT, variants=["raw", "size"],
                                     sample_policy="independent", market_cap_column="market_cap_proxy")
        pd.testing.assert_frame_equal(baseline.reset_index(drop=True),
                                      signals[signals.variant.eq("raw")].reset_index(drop=True), check_exact=True)
        size = signals[signals.variant.eq("size")]
        self.assertFalse(size.code.isin(bad).any())
        self.assertTrue(size.groupby("entry").size().eq(16).all())
        with self.assertRaisesRegex(ValueError, "complete common sample"):
            prepare_signals(frame, DIRECT, variants=["raw", "size"],
                            sample_policy="common", market_cap_column="market_cap_proxy")

    def test_missing_proxy_column_skips_size_without_suppressing_raw(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=12)
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["raw", "size"],
                                               sample_policy="independent", market_cap_column="market_cap_proxy")
        self.assertEqual(set(signals.variant), {"raw"})
        self.assertEqual(len(signals), len(frame))
        failed = diagnostics[diagnostics.variant.eq("size")].iloc[0]
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.reason, "missing_control_columns:market_cap_proxy")

    def test_constant_cap_is_redundant_instead_of_a_failed_cross_section(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=8)
        frame = frame.drop(columns=["industry", "float_mcap", *RISK_FIELDS])
        frame["market_cap_proxy"] = 1e10
        frame["value"] = np.arange(8, dtype=float)
        signals, diagnostics = prepare_signals(frame, DIRECT, variants=["size"],
                                               market_cap_column="market_cap_proxy")
        np.testing.assert_allclose(signals.signal, frame.value - frame.value.mean(), atol=1e-12)
        diagnostic = diagnostics.iloc[0]
        self.assertEqual(diagnostic["rank"], 1)
        self.assertEqual(diagnostic["columns"], 2)
        self.assertIn("log_market_cap_proxy", diagnostic.redundant_columns)

    def test_control_matrix_rejects_nonpositive_or_nonfinite_cap_explicitly(self):
        frame, _ = signal_fixture(weeks_count=1, stocks=4)
        for value in (0., -1., np.inf, np.nan):
            with self.subTest(value=value):
                invalid = frame.copy()
                invalid.loc[0, "float_mcap"] = value
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    control_matrix(invalid)

    def test_default_actual_cap_and_explicit_column_are_identical(self):
        frame, _ = signal_fixture(weeks_count=2, stocks=80)
        legacy, old_diagnostics = prepare_signals(frame)
        explicit, diagnostics = prepare_signals(frame, market_cap_column="float_mcap")
        pd.testing.assert_frame_equal(legacy, explicit, check_exact=True)
        pd.testing.assert_frame_equal(old_diagnostics, diagnostics, check_exact=True)


if __name__ == "__main__":
    unittest.main()
