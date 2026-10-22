from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.audit import audit
from fundamental_factor.config import DataGateError, load_config
from fundamental_factor.data import read_frame
from fundamental_factor.gf_adapter import INDUSTRIES, combine_raw_quotes, industry_candidates, status_from_quotes
from fundamental_factor.panel import build_panel
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture, enable_execution_fixture


class GFAdapterTests(unittest.TestCase):
    def test_suspension_and_unknown_limits_are_not_trade_permissions(self):
        quotes = pd.DataFrame({"date": pd.Timestamp("2025-01-02"), "code": ["000001.SZ"] * 4,
                               "open": [10., 10., 10., 10.], "close": [10., 0., 10., 10.],
                               "volume": [0., 2., np.nan, 3.]})
        status = status_from_quotes(quotes)
        self.assertEqual(status.suspended.iloc[:2].tolist(), [True, True])
        self.assertTrue(pd.isna(status.suspended.iloc[2]))
        self.assertFalse(status.suspended.iloc[3])
        self.assertTrue(status.limit_up.isna().all())
        quotes["limit_high"], quotes["limit_low"] = 10., 9.
        status = status_from_quotes(quotes)
        self.assertTrue(status.limit_up.all())
        self.assertFalse(status.limit_down.any())

    def test_raw_overlap_uses_keys_and_never_assumes_row_order(self):
        old = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                           "code": ["000001.SZ", "000002.SZ"], "open": [10., 20.],
                           "close": [11., 21.], "volume": [100., 200.]})
        new = old.iloc[::-1].copy()
        new.loc[new.code.eq("000001.SZ"), "close"] = 12.
        out, stats = combine_raw_quotes(old, new)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.set_index("code").loc["000001.SZ", "close"], 12.)
        self.assertEqual(stats["close_disagreements"], 1)
        with self.assertRaises(ValueError):
            combine_raw_quotes(pd.concat([old, old]), new)

    def test_industry_and_st_hints_cannot_masquerade_as_complete_history(self):
        source = pd.DataFrame({"date": [pd.Timestamp("2025-01-02")] * 3,
                               "code": ["000001.SZ", "000002.SZ", "000003.SZ"],
                               "secShortName": ["*ST测试", "普通", "未知"]})
        for column in INDUSTRIES:
            source[column] = 0.
        source.loc[0:1, INDUSTRIES[0]] = 1.
        source.loc[1, INDUSTRIES[1]] = 1.
        result = industry_candidates(source)
        self.assertEqual(result.industry.notna().tolist(), [True, False, False])
        self.assertNotIn("is_st", result)
        self.assertNotIn("float_mcap", result)


class ExecutionIntegrationTests(unittest.TestCase):
    def test_pipeline_unrestricted_execution_matches_theoretical_groups(self):
        with tempfile.TemporaryDirectory() as root:
            config = enable_execution_fixture(disk_fixture(Path(root), weeks_count=4))
            result = run_research(config)
            self.assertTrue(result["factors"][0]["execution"]["enabled"])
            directory = config.output / "factors" / "ROIC_new"
            actual = pd.read_csv(directory / "execution_weekly.csv")
            theory = pd.read_csv(directory / "portfolio_weekly.csv")
            keys = ["entry", "exit", "variant", "portfolio"]
            merged = actual.merge(theory, on=keys, suffixes=("_actual", "_theory"))
            # Price-only, unconstrained ledger must reproduce the frozen EW target.
            self.assertEqual(len(merged), len(actual))
            np.testing.assert_allclose(merged.return_actual, merged.return_theory, atol=1e-12)
            holdings = pd.read_parquet(directory / "execution_holdings.parquet")
            np.testing.assert_allclose(holdings.actual_weight, holdings.target_weight, atol=1e-12)
            self.assertTrue((pd.read_parquet(directory / "execution_daily.parquet").cash >= 0).all())

    def test_partial_events_block_even_when_file_exists_and_is_empty(self):
        with tempfile.TemporaryDirectory() as root:
            config = enable_execution_fixture(disk_fixture(Path(root), weeks_count=3))
            raw = json.loads(config.path.read_text())
            raw["execution"]["corporate_actions_complete"] = False
            config.path.write_text(json.dumps(raw))
            result = audit(load_config(config.path))
            self.assertFalse(result["gate_passed"])
            self.assertTrue(any("公司行动完整性" in message for message in result["blocking"]))

    def test_declared_coverage_must_include_final_exit(self):
        with tempfile.TemporaryDirectory() as root:
            config = enable_execution_fixture(disk_fixture(Path(root), weeks_count=3))
            raw = json.loads(config.path.read_text())
            raw["execution"]["coverage_end"] = "2015-01-19"
            config.path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(DataGateError, "覆盖期"):
                build_panel(load_config(config.path))

    def test_nullable_execution_status_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "status.parquet"
            pd.DataFrame({"date": [20250102], "code": ["000001.SZ"], "suspended": [False],
                          "limit_up": [None], "limit_down": [None]}).to_parquet(path)
            read = read_frame(path, nullable_booleans=True)
            self.assertTrue(read.limit_up.isna().all())
            with self.assertRaises(ValueError):
                read_frame(path)


if __name__ == "__main__":
    unittest.main()
