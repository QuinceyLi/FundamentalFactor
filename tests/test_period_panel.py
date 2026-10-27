"""Frequency and independent-sample acceptance tests on synthetic inputs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import DataGateError, RISK_FIELDS, load_config
from fundamental_factor.panel import build_panel, ensure_panel, rebalance_calendar, weekly_calendar
from tests.fixtures import disk_fixture


def independent_config(config, *, frequency="weekly", variants=None):
    raw = json.loads(config.path.read_text())
    raw["data_policy"] = {"mode": "classic", "missing_returns": "available_case", "empty_groups": "cash"}
    raw["backtest"]["rebalance_frequency"] = frequency
    raw["preprocess"] = {"winsorize": False, "standardize": False}
    raw["neutralization"] = {"include_float_mcap": False, "sample_policy": "independent",
                             "variants": variants or ["raw", "industry", "full"]}
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


class PeriodCalendarTests(unittest.TestCase):
    def test_monthly_holidays_and_partial_range_use_real_first_sessions(self):
        # Synthetic exchange closures cover both a month start and New Year.
        calendar = pd.Series(pd.bdate_range("2024-12-02", "2025-04-08"))
        closures = pd.to_datetime(["2025-01-01", "2025-02-03", "2025-02-04"])
        calendar = calendar[~calendar.isin(closures)]
        periods = rebalance_calendar(calendar, "2024-12-15", "2025-04-07", "monthly")
        self.assertEqual(periods.entry.tolist(), list(pd.to_datetime(["2025-01-02", "2025-02-05", "2025-03-03"])))
        self.assertEqual(periods.exit.tolist(), list(pd.to_datetime(["2025-02-05", "2025-03-03", "2025-04-01"])))
        self.assertTrue((periods.cutoff == periods.entry - pd.Timedelta(nanoseconds=1)).all())

    def test_month_end_range_does_not_create_an_incomplete_holding_period(self):
        calendar = pd.Series(pd.bdate_range("2025-01-01", "2025-05-01"))
        periods = rebalance_calendar(calendar, "2025-01-16", "2025-04-30", "monthly")
        self.assertEqual(periods.entry.tolist(), list(pd.to_datetime(["2025-02-03", "2025-03-03"])))
        self.assertEqual(periods.exit.iloc[-1], pd.Timestamp("2025-04-01"))
        with self.assertRaisesRegex(DataGateError, "月度"):
            rebalance_calendar(calendar, "2025-02-04", "2025-03-31", "monthly")

    def test_weekly_alias_preserves_existing_grid(self):
        calendar = pd.Series(pd.bdate_range("2025-01-01", "2025-03-04"))
        pd.testing.assert_frame_equal(weekly_calendar(calendar, "2025-01-08", "2025-03-04"),
                                      rebalance_calendar(calendar, "2025-01-08", "2025-03-04", "weekly"))
        with self.assertRaisesRegex(ValueError, "frequency"):
            rebalance_calendar(calendar, "2025-01-01", "2025-03-04", "daily")


class IndependentPanelTests(unittest.TestCase):
    def test_monthly_panel_uses_precomputed_values_and_monthly_forward_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = independent_config(disk_fixture(Path(tmp), weeks_count=22, stocks_count=12),
                                        frequency="monthly", variants=["raw"])
            source = pd.read_feather(config.source("roic"))
            snapshot_days = pd.to_datetime(["2015-01-30", "2015-02-27", "2015-03-31", "2015-04-30", "2015-05-29"])
            monthly = source[source.date.isin(snapshot_days)].copy()
            future = source[source.date.eq(pd.Timestamp("2015-02-02"))].copy()
            future["ROIC_new"] = 9999.
            pd.concat([monthly, future], ignore_index=True).to_feather(config.source("roic"))
            manifest = build_panel(config)
            periods = pd.read_parquet(config.output / "periods.parquet")
            pd.testing.assert_frame_equal(periods, pd.read_parquet(config.output / "weeks.parquet"))
            self.assertEqual(periods.entry.tolist(), list(pd.to_datetime(["2015-02-02", "2015-03-02", "2015-04-01", "2015-05-01"])))
            self.assertEqual(manifest["periods"], 4)
            self.assertEqual(manifest["rebalance_frequency"], "monthly")
            self.assertEqual(config.raw["metrics"]["annualization"], 12)
            self.assertEqual(manifest["calculation_version"], 4)
            self.assertIn("periods.parquet", manifest["artifact_fingerprints"])
            features = pd.read_parquet(config.output / "features.parquet")
            np.testing.assert_array_equal(features.signal, features.value)
            first = features[features.factor.eq("ROIC_new") & features.entry.eq(periods.entry.iloc[0])]
            expected = monthly[monthly.date.eq(pd.Timestamp("2015-01-30"))].set_index("code")["ROIC_new"]
            np.testing.assert_array_equal(first.set_index("code").signal.sort_index(), expected.sort_index())
            self.assertTrue(first.factor_date.eq(pd.Timestamp("2015-01-30")).all())
            self.assertTrue(features.factor_date.le(features.cutoff).all())
            labels = pd.read_parquet(config.output / "labels.parquet")
            quotes = pd.read_feather(config.source("quotes")).set_index(["date", "code"]).acc_open
            entry = pd.MultiIndex.from_frame(labels[["entry", "code"]])
            exit_ = pd.MultiIndex.from_frame(labels[["exit", "code"]])
            expected_returns = quotes.reindex(exit_).to_numpy() / quotes.reindex(entry).to_numpy() - 1
            np.testing.assert_allclose(labels["return"], expected_returns)
            self.assertEqual(ensure_panel(config)["calculation_version"], 4)

    def test_missing_controls_do_not_change_raw_signal_or_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = independent_config(disk_fixture(Path(tmp), weeks_count=3, stocks_count=40))
            history = pd.read_parquet(config.source("history"))
            history.loc[history.code.eq("000010.SZ"), "industry"] = None
            history.to_parquet(config.source("history"), index=False)
            risk = pd.read_feather(config.source("risk"))
            risk.loc[risk.code.eq("000011.SZ"), RISK_FIELDS[0]] = np.nan
            risk.to_feather(config.source("risk"))
            roic = pd.read_feather(config.source("roic"))
            roic.loc[roic.code.eq("000012.SZ"), "ROIC_new"] = np.nan
            roic.to_feather(config.source("roic"))
            build_panel(config)
            combined = pd.read_parquet(config.output / "features.parquet")
            roic_features = combined[combined.factor.eq("ROIC_new")]
            counts = roic_features.groupby(["entry", "variant"]).size().unstack()
            self.assertTrue(counts.raw.eq(39).all())
            self.assertTrue(counts.industry.eq(38).all())
            self.assertTrue(counts.full.eq(37).all())
            coverage = pd.read_csv(config.output / "coverage.csv")
            coverage = coverage[coverage.factor.eq("ROIC_new")]
            self.assertTrue(coverage.variant_raw_count.eq(39).all())
            self.assertTrue(coverage.variant_full_count.eq(37).all())
            self.assertTrue(coverage.common_count.eq(37).all())
            self.assertTrue(pd.read_csv(config.output / "benchmark.csv").n_stocks.eq(40).all())
            original = roic_features[roic_features.variant.eq("raw")][["entry", "code", "value", "signal", "percentile", "group"]]
            raw_config = json.loads(config.path.read_text())
            raw_config["output_dir"] = str(Path(tmp) / "raw_only")
            raw_config["neutralization"]["variants"] = ["raw"]
            config.path.write_text(json.dumps(raw_config))
            raw_config = load_config(config.path)
            build_panel(raw_config)
            alone = pd.read_parquet(raw_config.output / "features.parquet")
            alone = alone[alone.factor.eq("ROIC_new")][original.columns]
            pd.testing.assert_frame_equal(original.reset_index(drop=True), alone.reset_index(drop=True))

    def test_unidentifiable_neutral_version_keeps_raw_and_reports_its_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = independent_config(disk_fixture(Path(tmp), weeks_count=2, stocks_count=6))
            build_panel(config)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertEqual(set(features.variant), {"raw"})
            np.testing.assert_array_equal(features.signal, features.value)
            diagnostics = pd.read_csv(config.output / "preprocessing_failures.csv")
            failed = diagnostics[diagnostics.variant.eq("full")]
            self.assertTrue(failed.status.eq("failed").all())
            self.assertTrue(failed.reason.eq("insufficient_cross_section_degrees_of_freedom").all())
            coverage = pd.read_csv(config.output / "coverage.csv")
            self.assertTrue(coverage.variant_raw_count.eq(6).all())
            self.assertTrue(coverage.variant_full_count.eq(0).all())

    def test_missing_control_files_or_columns_leave_raw_available(self):
        for missing in ("risk_file", "control_columns"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                config = independent_config(disk_fixture(Path(tmp), weeks_count=2, stocks_count=40))
                if missing == "risk_file":
                    config.source("risk").unlink()
                else:
                    pd.read_feather(config.source("risk"))[["date", "code"]].to_feather(config.source("risk"))
                    pd.read_parquet(config.source("history")).drop(columns=["industry", "float_mcap"]).to_parquet(
                        config.source("history"), index=False)
                build_panel(config)
                signals = pd.read_parquet(config.output / "features.parquet")
                raw = signals[signals.variant.eq("raw")]
                self.assertEqual(len(raw), 2 * 2 * 40)
                np.testing.assert_array_equal(raw.signal, raw.value)
                self.assertFalse(signals.variant.eq("full").any())

    def test_new_settings_cannot_reuse_a_version_three_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = independent_config(disk_fixture(Path(tmp), weeks_count=2, stocks_count=6), variants=["raw"])
            manifest = build_panel(config)
            manifest["calculation_version"] = 3
            (config.output / "panel_manifest.json").write_text(json.dumps(manifest, default=str))
            with self.assertRaisesRegex(DataGateError, "版本"):
                ensure_panel(config)

    def test_newly_available_controls_invalidate_missing_source_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = independent_config(disk_fixture(Path(tmp), weeks_count=2, stocks_count=40))
            risk_path = config.source("risk")
            original_risk = pd.read_feather(risk_path)
            risk_path.unlink()
            manifest = build_panel(config)
            self.assertEqual(manifest["missing_source_paths"]["risk"], str(risk_path))
            ensure_panel(config)
            original_risk.to_feather(risk_path)
            with self.assertRaisesRegex(DataGateError, "已补齐"):
                ensure_panel(config)


if __name__ == "__main__":
    unittest.main()
