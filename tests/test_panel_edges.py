"""Focused regressions for data contracts and weekly sample construction."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_factor.audit import audit, source_columns
from fundamental_factor.config import DataGateError, load_config
from fundamental_factor.panel import build_panel, weekly_calendar
from fundamental_factor.preprocess import prepare_signals, winsorize
from tests.fixtures import disk_fixture, signal_fixture


class PanelEdgeTests(unittest.TestCase):
    def test_start_in_middle_of_week_does_not_invent_rebalance_date(self):
        calendar = pd.Series(pd.bdate_range("2025-01-01", "2025-02-03"))
        weeks = weekly_calendar(calendar, pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-27"))
        self.assertEqual(weeks.entry.iloc[0], pd.Timestamp("2025-01-13"))
        self.assertEqual(weeks.exit.iloc[-1], pd.Timestamp("2025-01-27"))
        self.assertFalse(weeks.entry.eq(pd.Timestamp("2025-01-08")).any())

    def test_mad_scale_and_zero_mass_quantile_fallback(self):
        clipped, method = winsorize(np.array([-1000., -2., -1., 0., 1., 2., 1000.]))
        self.assertEqual(method, "mad")
        self.assertAlmostEqual(clipped[0], -5 * 1.4826 * 2)
        self.assertAlmostEqual(clipped[-1], 5 * 1.4826 * 2)
        values = np.r_[np.zeros(95), np.ones(5)]
        clipped, method = winsorize(values)
        self.assertEqual(method, "quantile_fallback")
        self.assertTrue(np.all(clipped[:95] == 0))

    def test_full_neutral_failure_skips_all_variants_for_that_week(self):
        frame, weeks = signal_fixture(weeks_count=2, stocks=100)
        first = frame.entry.min()
        keep = frame.entry.ne(first) | frame.code.isin(frame.loc[frame.entry.eq(first), "code"].head(20))
        signals, failures = prepare_signals(frame[keep])
        self.assertFalse(signals.entry.eq(first).any())
        self.assertTrue(failures.loc[failures.entry.eq(first), "status"].eq("failed").all())
        self.assertEqual(set(signals.variant), {"raw", "industry_size", "full"})
        self.assertTrue(signals.entry.eq(weeks.entry.iloc[1]).all())
        for _, group in signals.groupby("variant"):
            self.assertEqual(len(group), 100)

    def test_missing_industry_and_cap_remove_signals_but_not_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            history = pd.read_parquet(config.source("history"))
            history.loc[history.code.eq("000010.SZ"), ["industry", "float_mcap"]] = [None, np.nan]
            history.to_parquet(config.source("history"), index=False)
            self.assertTrue(audit(config)["gate_passed"])
            build_panel(config)
            signals = pd.read_parquet(config.output / "features.parquet")
            self.assertFalse(signals.code.eq("000010.SZ").any())
            labels = pd.read_parquet(config.output / "labels.parquet")
            self.assertTrue(labels.code.eq("000010.SZ").any())
            self.assertTrue(pd.read_csv(config.output / "benchmark.csv").n_stocks.eq(100).all())

    def test_missing_current_st_history_stops_base_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            history = pd.read_parquet(config.source("history"))
            history.loc[history.code.eq("000003.SZ"), "date"] = pd.Timestamp("2015-01-06")
            history.to_parquet(config.source("history"), index=False)
            self.assertTrue(audit(config)["gate_passed"])
            with self.assertRaisesRegex(DataGateError, "ST"):
                build_panel(config)

    def test_missing_entry_suspension_status_stops_base_universe(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            status = pd.read_parquet(config.source("status"))
            status = status[~(status.date.eq(pd.Timestamp("2015-01-05")) & status.code.eq("000003.SZ"))]
            status.to_parquet(config.source("status"), index=False)
            self.assertTrue(audit(config)["gate_passed"])
            with self.assertRaisesRegex(DataGateError, "停牌"):
                build_panel(config)

    def test_new_listing_can_be_excluded_before_st_and_status_are_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            code = "000003.SZ"
            stocks = pd.read_parquet(config.source("stocks"))
            stocks.loc[stocks.code.eq(code), "listing_date"] = pd.Timestamp("2015-01-02")
            stocks.to_parquet(config.source("stocks"), index=False)
            history = pd.read_parquet(config.source("history"))
            history = history[~history.code.eq(code)]
            history.to_parquet(config.source("history"), index=False)
            status = pd.read_parquet(config.source("status"))
            status = status[~status.code.eq(code)]
            status.to_parquet(config.source("status"), index=False)
            build_panel(config)
            self.assertFalse(pd.read_parquet(config.output / "labels.parquet").code.eq(code).any())

    def test_two_factors_can_share_one_source_file_and_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            roic = pd.read_feather(config.source("roic"))
            fund = pd.read_feather(config.source("fund"))
            roic.merge(fund, on=["date", "code"], validate="one_to_one").to_feather(config.source("roic"))
            raw = json.loads(config.path.read_text())
            raw["factors"][1]["source"] = "roic"
            config.path.write_text(json.dumps(raw))
            config = load_config(config.path)
            self.assertEqual(set(source_columns(config)["roic"]), {"date", "code", "ROIC_new", "fundnum_promotion"})
            self.assertTrue(audit(config)["gate_passed"])
            build_panel(config)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertEqual(set(features.factor), {"ROIC_new", "fundnum_promotion"})
            self.assertEqual(features.groupby("factor").variant.nunique().to_dict(),
                             {"ROIC_new": 3, "fundnum_promotion": 3})

    def test_a_share_flag_accepts_zero_one_and_excludes_explicit_non_a_stock(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            stocks = pd.read_parquet(config.source("stocks"))
            stocks["is_a_share"] = 1
            stocks.loc[stocks.code.eq("000003.SZ"), "is_a_share"] = 0
            stocks.to_parquet(config.source("stocks"), index=False)
            self.assertTrue(audit(config)["gate_passed"])
            build_panel(config)
            labels = pd.read_parquet(config.output / "labels.parquet")
            self.assertFalse(labels.code.eq("000003.SZ").any())
            self.assertTrue(pd.read_csv(config.output / "benchmark.csv").n_stocks.eq(99).all())

    def test_a_share_flag_missing_values_are_not_silent_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            stocks = pd.read_parquet(config.source("stocks"))
            stocks["is_a_share"] = stocks.is_a_share.astype("boolean")
            stocks.loc[stocks.code.eq("000003.SZ"), "is_a_share"] = pd.NA
            stocks.to_parquet(config.source("stocks"), index=False)
            checked = audit(config)
            self.assertFalse(checked["gate_passed"])
            self.assertTrue(any("stocks" in issue for issue in checked["blocking"]))


if __name__ == "__main__":
    unittest.main()
