from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fundamental_factor.config import DataGateError, load_config
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture, enable_execution_fixture


def classic(config, **updates):
    raw = json.loads(config.path.read_text())
    raw["data_policy"] = {"mode": "classic"}
    raw.update(updates)
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


class ClassicResearchTests(unittest.TestCase):
    def test_missing_optional_sources_and_labels_do_not_stop_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=3, stocks_count=40)
            sources = dict(config.raw["sources"])
            sources.update(history=None, risk=None, status=None, terminal=None)
            config = classic(config, sources=sources,
                             neutralization={"include_float_mcap": False, "variants": ["raw"]},
                             execution={"enabled": True})
            quotes = pd.read_feather(config.source("quotes"))
            quotes.loc[quotes.date.eq("2015-01-12") & quotes.code.eq("000003.SZ"), "acc_open"] = np.nan
            quotes.to_feather(config.source("quotes"))
            result = run_research(config)
            self.assertEqual(result["status"], "completed")
            audit = json.loads((config.output / "audit.json").read_text())
            self.assertTrue(audit["gate_passed"])
            self.assertFalse(audit["execution_gate_passed"])
            for card in result["factors"]:
                self.assertEqual(card["variants"], ["raw"])
                self.assertEqual(card["sample"]["signal_rows"], 120)
                self.assertEqual(card["execution"]["status"], "skipped")
                self.assertTrue(card["execution"]["blocking"])
                folder = config.output / card["output_dir"]
                weekly = pd.read_csv(folder / "portfolio_weekly.csv")
                self.assertTrue(weekly.query("portfolio == 'ALL'")["return"].isna().any())
                holdings = pd.read_parquet(folder / "holdings.parquet")
                missing_stock = holdings.query("portfolio == 'ALL' and code == '000003.SZ'")
                self.assertEqual(len(missing_stock), 3)
                self.assertTrue(missing_stock.target_weight.eq(1 / 40).all())
                self.assertFalse((folder / "execution_weekly.csv").exists())
            labels = pd.read_parquet(config.output / "labels.parquet")
            self.assertTrue(labels.trade_restriction_unknown.all())
            self.assertGreater(labels["return"].notna().sum(), 0)
            self.assertTrue(labels.loc[labels.unresolved, "return"].isna().all())

    def test_optional_runtime_failure_is_disclosed_without_losing_factor_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic(enable_execution_fixture(disk_fixture(Path(tmp), weeks_count=2)))
            with patch("fundamental_factor.execution.run_execution", side_effect=DataGateError("unresolved optional account")):
                result = run_research(config)
            self.assertEqual(result["status"], "completed")
            for card in result["factors"]:
                self.assertEqual(card["execution"]["status"], "skipped")
                self.assertIn("unresolved optional account", card["execution"]["blocking"])
                self.assertTrue((config.output / card["output_dir"] / "ic_summary.csv").exists())

    def test_industry_neutralization_runs_without_market_cap_or_barra(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=2, stocks_count=40)
            history = pd.read_parquet(config.source("history")).drop(columns=["float_mcap", "is_st"])
            history.to_parquet(config.source("history"), index=False)
            sources = dict(config.raw["sources"])
            sources["risk"] = None
            config = classic(config, sources=sources,
                             neutralization={"include_float_mcap": False, "variants": ["raw", "industry"]})
            result = run_research(config)
            for card in result["factors"]:
                self.assertEqual(set(card["summary"]), {"raw", "industry"})
                self.assertEqual(card["sample"]["signal_rows"], 80)


if __name__ == "__main__":
    unittest.main()
