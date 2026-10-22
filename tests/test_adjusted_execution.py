"""A proxy execution ledger uses adjusted units without inventing cash events."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_factor.audit import audit, source_columns
from fundamental_factor.config import load_config
from fundamental_factor.execution import run_execution
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture
from tests.test_execution import fixture
from tests.test_proxy_mode import proxy_config


def enable_adjusted_execution(config):
    raw = json.loads(config.path.read_text())
    raw["execution"] = {"enabled": True, "price_mode": "adjusted_units", "initial_capital": 100.,
                        "description": "SYNTHETIC adjusted units and explicitly simulated trading proxies."}
    raw["sources"].update(execution_status="status.parquet", corporate_actions=None, execution_terminal=None)
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


class AdjustedExecutionTests(unittest.TestCase):
    def test_adjusted_quantities_and_price_rescaling_do_not_change_weights_or_returns(self):
        inputs = fixture([["A", "B"], ["B", "C"], ["A", "C"]])
        week = inputs["weeks"].entry.iloc[1]
        prices = inputs["quotes"]
        prices.loc[prices.code.eq("B") & prices.date.ge(week), ["open", "close"]] = 11.
        reference = run_execution(**inputs)
        prices[["open", "close"]] = prices[["open", "close"]].mul(prices.code.map({"A": 10., "B": .3, "C": 2.}), axis=0)
        result = run_execution(**inputs, price_mode="adjusted_units")
        np.testing.assert_allclose(result["weekly"]["return"], reference["weekly"]["return"], atol=1e-12)
        np.testing.assert_allclose(result["weekly"].turnover, reference["weekly"].turnover, atol=1e-12)
        np.testing.assert_allclose(result["holdings"].actual_weight, reference["holdings"].actual_weight, atol=1e-12)
        self.assertIn("adjusted_units", result["holdings"])
        self.assertNotIn("shares", result["holdings"])
        self.assertIn("executed_units", result["trades"])
        self.assertTrue(result["weekly"].price_mode.eq("adjusted_units").all())

    def test_raw_share_cash_events_are_rejected_for_adjusted_units(self):
        for source in ("actions", "terminal"):
            with self.subTest(source=source):
                inputs = fixture([["A"]])
                day = pd.Timestamp("2026-01-07")
                record = dict(date=day, code="A", cash_per_share=1., share_ratio=0., notice_date=day, available_date=day)
                inputs[source] = pd.DataFrame([record])
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    run_execution(**inputs, price_mode="adjusted_units")

    def test_delisting_invalidates_only_accounts_that_actually_hold_it_from_that_week(self):
        inputs = fixture([["A", "B"], ["A", "B"], ["B"]])
        inputs["signals"].loc[inputs["signals"].code.eq("B"), "group"] = 1
        inputs["portfolios"] = ["ALL", "Q1", "Q10"]
        death = pd.Timestamp("2026-01-14")
        result = run_execution(**inputs, price_mode="adjusted_units",
                               delistings=pd.DataFrame({"code": ["A"], "delisting_date": [death]}))
        for portfolio in ("ALL", "Q10"):
            affected = result["weekly"].query("portfolio == @portfolio").sort_values("entry")
            self.assertEqual(affected.quality_status.iloc[0], "ok")
            self.assertTrue(affected["return"].iloc[1:].isna().all())
            self.assertTrue(affected.nav.iloc[1:].isna().all())
            self.assertTrue(affected.indicative_nav.notna().all())
            self.assertTrue(affected.quality_reason.iloc[1:].eq("unresolved_delisting:A").all())
            self.assertFalse(result["trades"].query("portfolio == @portfolio").entry.gt(death).any())
            performance = result["performance"].query("portfolio == @portfolio").iloc[0]
            self.assertTrue(np.isnan(performance.total_return))
        unaffected = result["weekly"].query("portfolio == 'Q1'")
        self.assertTrue(unaffected.quality_status.eq("ok").all())
        self.assertTrue(unaffected["return"].notna().all())
        daily = result["daily"].query("portfolio == 'Q10'")
        self.assertTrue(daily.loc[daily.date.ge(death), "nav"].isna().all())
        self.assertTrue(daily.loc[daily.date.lt(death), "nav"].notna().all())

    def test_delisted_unheld_code_does_not_invalidate_and_cannot_be_bought(self):
        inputs = fixture([["A", "B"]])
        death = inputs["weeks"].entry.iloc[0]
        result = run_execution(**inputs, price_mode="adjusted_units",
                               delistings=pd.DataFrame({"code": ["B"], "delisting_date": [death]}))
        self.assertTrue(result["weekly"].quality_status.eq("ok").all())
        rejected = result["trades"].query("code == 'B'").iloc[0]
        self.assertEqual(rejected.rejection_reason, "delisted_unresolved")
        self.assertEqual(rejected.executed_units, 0.)

    def test_unknown_limits_freeze_existing_units_and_preserve_cash_constraint(self):
        inputs = fixture([["A"], ["B"]])
        week = inputs["weeks"].entry.iloc[1]
        inputs["status"]["limit_down"] = inputs["status"].limit_down.astype("boolean")
        inputs["status"].loc[inputs["status"].code.eq("A") & inputs["status"].date.eq(week), "limit_down"] = pd.NA
        result = run_execution(**inputs, price_mode="adjusted_units")
        held = result["holdings"].query("entry == @week").set_index("code")
        self.assertEqual(held.loc["A", "adjusted_units"], 10.)
        self.assertEqual(held.loc["B", "adjusted_units"], 0.)
        self.assertTrue(result["daily"].cash.ge(0).all())
        reason = result["trades"].query("entry == @week and code == 'A'").rejection_reason.iloc[0]
        self.assertEqual(reason, "unknown_limit_status")

    def test_config_audit_does_not_require_raw_actions_but_prevents_double_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = enable_adjusted_execution(proxy_config(disk_fixture(Path(tmp), weeks_count=3)))
            required = source_columns(config)
            self.assertNotIn("corporate_actions", required)
            self.assertNotIn("execution_quotes", required)
            self.assertTrue(audit(config)["gate_passed"])
            raw = json.loads(config.path.read_text())
            raw["output_dir"] = str(Path(tmp) / "other_run")
            raw["sources"]["corporate_actions"] = "must_not_use.csv"
            config.path.write_text(json.dumps(raw))
            checked = audit(load_config(config.path))
            self.assertFalse(checked["gate_passed"])
            self.assertTrue(any("重复调整" in issue for issue in checked["blocking"]))

    def test_incomplete_execution_does_not_abort_factor_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = enable_adjusted_execution(proxy_config(disk_fixture(Path(tmp), weeks_count=3)))
            stocks = pd.read_parquet(config.source("stocks"))
            stocks.loc[stocks.code.eq("000003.SZ"), "delisting_date"] = pd.Timestamp("2015-01-08")
            stocks.to_parquet(config.source("stocks"), index=False)
            result = run_research(config)
            self.assertEqual(result["status"], "completed")
            for factor in result["factors"]:
                self.assertEqual(factor["execution"]["price_mode"], "adjusted_units")
                self.assertEqual(factor["execution"]["quality_status"], "incomplete")
                self.assertGreater(factor["execution"]["incomplete_account_weeks"], 0)
                directory = config.output / factor["output_dir"]
                self.assertTrue((directory / "ic_summary.csv").is_file())
                weekly = pd.read_csv(directory / "execution_weekly.csv")
                self.assertTrue(weekly.loc[weekly.quality_status.ne("ok"), "return"].isna().all())
                self.assertIn("adjusted_units", pd.read_parquet(directory / "execution_holdings.parquet"))


if __name__ == "__main__":
    unittest.main()
