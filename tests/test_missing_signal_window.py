"""Signal availability controls the test window without selecting on returns."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fundamental_factor.config import Config, load_config
from fundamental_factor.panel import build_panel
from fundamental_factor.report import build_report_data
from fundamental_factor.research import describe_evaluation_window, run_research
from tests.fixtures import disk_fixture


class EvaluationWindowTests(unittest.TestCase):
    def test_missing_years_partial_years_and_interior_gaps_are_disclosed(self):
        starts = pd.date_range("2017-01-01", "2021-04-01", freq="MS")
        weeks = pd.DataFrame({"entry": starts[:-1], "exit": starts[1:]})
        weeks["cutoff"] = weeks.entry - pd.Timedelta(nanoseconds=1)
        raw = weeks.loc[weeks.entry.ge("2018-11-01") & weeks.entry.ne("2020-07-01"), ["entry", "exit"]].assign(variant="raw")
        industry = weeks.loc[weeks.entry.ge("2019-04-01"), ["entry", "exit"]].assign(variant="industry")
        signals = pd.concat([raw, industry], ignore_index=True)
        excluded = pd.concat([weeks.loc[~weeks.entry.isin(block.entry), ["entry", "exit"]].assign(variant=variant)
                              for variant, block in [("raw", raw), ("industry", industry)]], ignore_index=True)
        config = Config({"neutralization": {"variants": ["raw", "industry"]}, "backtest": {
            "start_date": "2017-01-01", "end_date": "2021-03-31"}}, Path("unused"))
        got = describe_evaluation_window(config, weeks, signals, excluded)
        self.assertEqual(got["requested_periods"], 51)
        self.assertEqual(got["requested_end"], "2021-03-31")
        raw_window, industry_window = got["variants"]["raw"], got["variants"]["industry"]
        self.assertEqual(raw_window["holding_start"], "2018-11-01")
        self.assertEqual(raw_window["excluded_years"], [2017])
        self.assertEqual(raw_window["partially_excluded_years"], [2018, 2020])
        self.assertEqual(raw_window["segments"], 2)
        self.assertEqual(industry_window["excluded_years"], [2017, 2018])
        self.assertEqual(industry_window["partially_excluded_years"], [2019])
        self.assertEqual(industry_window["holding_end"], "2021-04-01")
        self.assertEqual(industry_window["segments"], 1)
        for row in got["variants"].values():
            self.assertEqual(row["included_periods"] + row["excluded_periods"], len(weeks))

    def test_research_exports_actual_variant_windows_and_report_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=4, stocks_count=40)
            raw = json.loads(config.path.read_text())
            raw["neutralization"] = {"variants": ["raw", "industry"], "sample_policy": "independent", "include_float_mcap": False}
            raw["data_policy"] = {"mode": "classic", "missing_returns": "available_case", "empty_groups": "cash"}
            config.path.write_text(json.dumps(raw))
            config = load_config(config.path)
            manifest = build_panel(config)
            grid = pd.read_parquet(config.output / "weeks.parquet")
            features = pd.read_parquet(config.output / "features.parquet")
            target = features.factor.eq("ROIC_new")
            missing = target & ((features.variant.eq("raw") & features.entry.lt(grid.entry.iloc[2])) |
                                (features.variant.eq("industry") & features.entry.lt(grid.entry.iloc[3])))
            features.loc[~missing].to_parquet(config.output / "features.parquet", index=False)
            labels = pd.read_parquet(config.output / "labels.parquet")
            missing_return = labels.entry.eq(grid.entry.iloc[-1]) & labels.code.eq("000000.SZ")
            labels.loc[missing_return, ["return", "unresolved"]] = [np.nan, True]
            labels.to_parquet(config.output / "labels.parquet", index=False)
            # Supply an intentionally sparse, already validated panel; do not
            # let the source builder regenerate the removed signal periods.
            with patch("fundamental_factor.research.ensure_panel", return_value=manifest):
                result = run_research(config)
            card = next(f for f in result["factors"] if f["name"] == "ROIC_new")
            self.assertEqual(card["sample"]["requested_periods"], 4)
            self.assertEqual(card["sample"]["periods"], 2)
            self.assertEqual(card["evaluation_window"]["variants"]["raw"]["included_periods"], 2)
            self.assertEqual(card["evaluation_window"]["variants"]["industry"]["included_periods"], 1)
            self.assertEqual(len(pd.read_parquet(config.output / "weeks.parquet")), 4)
            folder = config.output / card["output_dir"]
            excluded = pd.read_csv(folder / "excluded_periods.csv")
            self.assertEqual(excluded.groupby("variant").size().to_dict(), {"raw": 2, "industry": 3})
            portfolio = pd.read_csv(folder / "portfolio_weekly.csv")
            self.assertEqual(portfolio.query("portfolio == 'ALL'").groupby("variant").size().to_dict(), {"raw": 2, "industry": 1})
            # A missing stock label still has its frozen target holding.
            holdings = pd.read_parquet(folder / "holdings.parquet")
            held = holdings.loc[holdings.code.eq("000000.SZ") & holdings.entry.eq(grid.entry.iloc[-1]) & holdings.portfolio.eq("ALL")]
            self.assertTrue(held["return"].isna().all())
            self.assertTrue(held.target_weight.gt(0).all())
            payload = build_report_data(config.output)
            self.assertFalse(payload["issues"])
            report_factor = next(f for f in payload["factors"] if f["name"] == "ROIC_new")
            self.assertEqual(report_factor["evaluation_window"], card["evaluation_window"])
            self.assertEqual(len(report_factor["excluded_periods"]), 5)
            self.assertTrue(any(link["name"].endswith("excluded_periods.csv") for link in payload["links"]))


if __name__ == "__main__":
    unittest.main()
