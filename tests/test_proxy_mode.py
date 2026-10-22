"""Proxy contracts preserve uncertainty and frozen samples instead of inventing data."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_factor.audit import audit
from fundamental_factor.backtest import evaluate_factor
from fundamental_factor.config import DataGateError, load_config
from fundamental_factor.panel import base_universe, build_panel, ensure_panel, make_labels, weekly_calendar
from tests.fixtures import disk_fixture
from tests.test_backtest_edges import _fixture


def proxy_config(config, **overrides):
    raw = json.loads(config.path.read_text())
    raw["data_policy"] = {"mode": "proxy", "st_unknown": "include", "suspension_unknown": "exclude",
                          "missing_returns": "report", "description": "SYNTHETIC proxy acceptance scenario."}
    raw["data_policy"].update(overrides)
    raw["price_contract"].update(total_return=False, suspended_total_return=False,
                                  return_kind="adjusted_price", suspension_valuation="previous_adjusted_close")
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


class ProxyModeTests(unittest.TestCase):
    def test_classic_default_and_explicit_legacy_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=3)
            raw = json.loads(config.path.read_text())
            raw["data_policy"] = {"missing_returns": "report"}
            config.path.write_text(json.dumps(raw))
            self.assertEqual(load_config(config.path).data_policy["mode"], "classic")
            raw["data_policy"]["mode"] = "strict"
            config.path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "mode=proxy"):
                load_config(config.path)
            raw["data_policy"] = {"mode": "proxy", "description": ""}
            config.path.write_text(json.dumps(raw))
            checked = audit(load_config(config.path))
            self.assertFalse(checked["gate_passed"])
            self.assertTrue(any("代理研究必须" in value for value in checked["blocking"]))

    def test_nullable_st_is_preserved_known_st_wins_and_future_is_not_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = proxy_config(disk_fixture(Path(tmp), weeks_count=3))
            history = pd.read_parquet(config.source("history"))
            history["is_st"] = history.is_st.astype("boolean")
            history.loc[history.code.eq("000003.SZ"), "is_st"] = pd.NA
            history.loc[history.code.eq("000004.SZ"), "is_st"] = True
            history.loc[history.code.eq("000005.SZ"), ["date", "is_st"]] = [pd.Timestamp("2015-01-06"), True]
            history.to_parquet(config.source("history"), index=False)
            self.assertTrue(audit(config)["gate_passed"])
            weeks = weekly_calendar(pd.Series(pd.bdate_range("2015-01-02", "2015-01-26")),
                                    pd.Timestamp("2015-01-05"), pd.Timestamp("2015-01-26"))
            base = base_universe(config, weeks)
            unknown = base[base.code.eq("000003.SZ")]
            self.assertTrue(unknown.is_st.isna().all())
            self.assertTrue(unknown.is_st_unknown.all())
            self.assertFalse(base.code.eq("000004.SZ").any())
            future = base[base.code.eq("000005.SZ")]
            self.assertEqual(future.entry.tolist(), [pd.Timestamp("2015-01-05")])
            self.assertTrue(future.is_st_unknown.all())

    def test_unknown_st_exclusion_is_a_signal_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = proxy_config(disk_fixture(Path(tmp), weeks_count=3), st_unknown="exclude")
            history = pd.read_parquet(config.source("history"))
            history = history[~history.code.eq("000003.SZ")]
            history.to_parquet(config.source("history"), index=False)
            build_panel(config)
            self.assertFalse(pd.read_parquet(config.output / "features.parquet").code.eq("000003.SZ").any())
            universe = pd.read_parquet(config.output / "universe.parquet")
            self.assertTrue(universe.loc[universe.code.eq("000003.SZ"), "is_st_unknown"].all())

    def test_unknown_suspension_is_excluded_only_at_that_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = proxy_config(disk_fixture(Path(tmp), weeks_count=3))
            status = pd.read_parquet(config.source("status"))
            status = status[~(status.date.eq("2015-01-12") & status.code.eq("000003.SZ"))]
            status.to_parquet(config.source("status"), index=False)
            build_panel(config)
            universe = pd.read_parquet(config.output / "universe.parquet")
            stock = universe[universe.code.eq("000003.SZ")].set_index("entry")
            self.assertTrue(stock.loc[pd.Timestamp("2015-01-05"), "eligible"])
            self.assertFalse(stock.loc[pd.Timestamp("2015-01-12"), "eligible"])
            self.assertTrue(stock.loc[pd.Timestamp("2015-01-12"), "entry_suspension_unknown"])
            labels = pd.read_parquet(config.output / "labels.parquet")
            first = labels[labels.entry.eq("2015-01-05") & labels.code.eq("000003.SZ")].iloc[0]
            self.assertTrue(first.unresolved)
            self.assertIn("exit_status_missing", first.unresolved_reason)

    def test_missing_return_retains_weights_ic_pairs_and_breaks_turnover_only_when_unknown(self):
        signals, labels, benchmark, weeks = _fixture(first_low=.01, first_high=.1)
        missing = labels.entry.eq(weeks.entry.iloc[0]) & labels.code.eq("000005.SZ")
        labels.loc[missing, ["return", "unresolved"]] = [np.nan, True]
        benchmark.loc[0, "return"] = np.nan
        before = signals.copy(deep=True)
        results = evaluate_factor(signals, labels, benchmark, weeks, missing_returns="report")
        pd.testing.assert_frame_equal(signals, before)
        first_ic = results["ic_weekly"].query("subset == 'ALL' and method == 'spearman'").iloc[0]
        self.assertEqual(first_ic.n_stocks, 10)
        self.assertEqual(first_ic.n_labels, 9)
        self.assertAlmostEqual(first_ic.label_coverage, .9)
        q6 = results["portfolio_weekly"].query("portfolio == 'Q6'").sort_values("entry")
        self.assertTrue(np.isnan(q6["return"].iloc[0]))
        self.assertEqual(q6.reason.iloc[0], "missing_held_return")
        self.assertEqual(q6.turnover.iloc[0], 1.)
        self.assertTrue(np.isnan(q6.turnover.iloc[1]))
        self.assertEqual(q6.turnover.iloc[2], 0.)
        ledger = results["holdings"].query("portfolio == 'Q6'").sort_values("entry")
        self.assertTrue(ledger.target_weight.eq(1.).all())
        self.assertTrue(np.isnan(ledger.drifted_weight.iloc[1]))
        q7 = results["portfolio_weekly"].query("portfolio == 'Q7'")
        self.assertTrue(q7["return"].notna().all())
        perf = results["performance"].query("portfolio == 'Q6'").iloc[0]
        self.assertTrue(np.isnan(perf.total_return))
        with self.assertRaises(DataGateError):
            evaluate_factor(signals, labels, benchmark, weeks)

    def test_absent_label_row_is_not_removed_or_reweighted(self):
        signals, labels, benchmark, weeks = _fixture(first_low=.01, first_high=.1)
        results = evaluate_factor(signals, labels.iloc[1:], benchmark, weeks, missing_returns="report")
        all_first = results["portfolio_weekly"].query("portfolio == 'ALL'").iloc[0]
        self.assertEqual(all_first.n_stocks, 10)
        self.assertEqual(all_first.n_labels, 9)
        self.assertTrue(np.isnan(all_first["return"]))
        first = results["holdings"].query("portfolio == 'ALL'")
        first = first[first.entry.eq(weeks.entry.iloc[0])]
        self.assertEqual(len(first), 10)
        self.assertTrue(first.target_weight.eq(.1).all())

    def test_adjusted_proxy_carry_and_unresolved_delisting_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = proxy_config(disk_fixture(Path(tmp), weeks_count=3))
            status = pd.read_parquet(config.source("status"))
            status.loc[status.date.eq("2015-01-12") & status.code.eq("000003.SZ"), "suspended"] = True
            status.to_parquet(config.source("status"), index=False)
            features = pd.DataFrame({"entry": [pd.Timestamp("2015-01-05")],
                                     "exit": [pd.Timestamp("2015-01-12")], "code": ["000003.SZ"],
                                     "delisting_date": [pd.NaT]})
            labels = make_labels(config, features)
            self.assertFalse(labels.unresolved.iloc[0])
            self.assertTrue(labels.exit_suspended.iloc[0])
            self.assertTrue(labels.trade_restricted.iloc[0])
            features["delisting_date"] = pd.Timestamp("2015-01-09")
            labels = make_labels(config, features)
            self.assertTrue(labels.unresolved.iloc[0])
            self.assertTrue(np.isnan(labels["return"].iloc[0]))
            self.assertEqual(labels.unresolved_reason.iloc[0], "delisting_without_terminal")

    def test_pipeline_keeps_incomplete_labels_without_claiming_complete_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = proxy_config(disk_fixture(Path(tmp), weeks_count=3))
            prices = pd.read_feather(config.source("quotes"))
            prices.loc[prices.date.eq("2015-01-12") & prices.code.eq("000003.SZ"), "acc_open"] = np.nan
            prices.to_feather(config.source("quotes"))
            manifest = build_panel(config)
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["quality_status"], "unresolved_labels")
            self.assertGreater(manifest["unresolved_labels"], 0)
            self.assertGreater(manifest["incomplete_benchmark_weeks"], 0)
            self.assertEqual(ensure_panel(config)["status"], "ready")
            signals = pd.read_parquet(config.output / "features.parquet")
            first = signals[signals.entry.eq("2015-01-05")]
            self.assertTrue(first.code.eq("000003.SZ").any())
            labels = pd.read_parquet(config.output / "labels.parquet")
            self.assertTrue(labels.loc[labels.unresolved, "return"].isna().all())


if __name__ == "__main__":
    unittest.main()
