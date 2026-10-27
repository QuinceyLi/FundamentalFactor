"""The optional size control must not become a gate on independent raw factors."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fundamental_factor.audit import audit, source_columns
from fundamental_factor.config import load_config
from tests.fixtures import disk_fixture


class MarketCapConfigAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = disk_fixture(Path(self.temp.name), weeks_count=3, stocks_count=12)
        self.raw = json.loads(self.config.path.read_text(encoding="utf-8"))

    def load(self):
        self.config.path.write_text(json.dumps(self.raw), encoding="utf-8")
        return load_config(self.config.path)

    def proxy(self, *, variants=None, shared=False, optional=True):
        self.raw["data_policy"] = {"mode": "classic"}
        self.raw["neutralization"] = {
            "variants": variants or ["raw", "size"], "include_float_mcap": False,
            "sample_policy": "independent" if optional else "common",
            "market_cap": {"method": "quote_proxy"},
        }
        quotes = pd.read_feather(self.config.source("quotes"))
        quotes["turnover"] = 1.5
        if shared:
            quotes["close"], quotes["volume"] = 10., 15000.
            self.raw["neutralization"]["market_cap"]["quotes_source"] = "quotes"
        else:
            self.raw["sources"]["market_cap_quotes"] = "raw_quotes.parquet"
            raw_quotes = quotes[["date", "code"]].assign(close=10., volume=15000.)
            raw_quotes.to_parquet(Path(self.raw["data_root"]) / "raw_quotes.parquet", index=False)
        quotes.to_feather(self.config.source("quotes"))
        return self.load()

    def test_legacy_defaults_and_digest_do_not_change_when_reading_new_properties(self):
        previous = copy.deepcopy(self.config.raw)
        digest = self.config.digest
        self.assertEqual(self.config.market_cap["method"], "history")
        self.assertEqual(self.config.market_cap_column, "float_mcap")
        self.assertTrue(self.config.requires_market_cap)
        self.assertTrue(self.config.requires_float_mcap)
        self.assertFalse(self.config.uses_market_cap_proxy)
        self.assertEqual(self.config.raw, previous)
        self.assertEqual(self.config.digest, digest)
        self.assertNotIn("neutralization", self.config.raw)

    def test_size_requires_cap_even_when_full_cap_switch_is_false(self):
        config = self.proxy()
        self.assertTrue(config.requires_market_cap)
        self.assertTrue(config.uses_market_cap_proxy)
        self.assertFalse(config.requires_float_mcap)
        self.assertFalse(config.requires_industry)
        self.assertFalse(config.requires_risk)
        self.assertEqual(config.market_cap_column, "market_cap_proxy")
        self.assertEqual(source_columns(config)["quotes"], ["date", "code", "acc_open", "acc_close", "turnover"])
        self.assertEqual(source_columns(config)["market_cap_quotes"], ["date", "code", "close", "volume"])
        self.raw["neutralization"]["variants"] = ["full"]
        config = self.load()
        self.assertTrue(config.requires_industry)
        self.assertTrue(config.requires_risk)
        self.assertFalse(config.requires_market_cap)
        self.assertFalse(config.uses_market_cap_proxy)

    def test_proxy_audit_needs_neither_history_industry_nor_float_cap(self):
        config = self.proxy(variants=["size"], optional=False)
        self.raw["sources"]["history"] = None
        self.raw["sources"]["risk"] = None
        result = audit(self.load())
        self.assertTrue(result["gate_passed"], result["blocking"])
        self.assertTrue(result["market_cap"]["is_proxy"])
        self.assertEqual(result["market_cap"]["formula"], "close * volume * 100 / (turnover / 100)")
        self.assertEqual(result["sources"]["market_cap_quotes"]["numeric"]["volume"]["min"], 15000.)

    def test_shared_proxy_source_preserves_core_adjusted_prices(self):
        config = self.proxy(shared=True)
        self.assertEqual(source_columns(config)["quotes"],
                         ["date", "code", "acc_open", "acc_close", "close", "volume", "turnover"])
        quotes = pd.read_feather(config.source("quotes")).drop(columns=["volume", "turnover"])
        quotes.to_feather(config.source("quotes"))
        result = audit(config)
        self.assertTrue(result["gate_passed"], result["blocking"])
        self.assertTrue(any("缺少市值代理字段" in value for value in result["warnings"]))
        self.assertEqual(result["sources"]["quotes"]["selected_columns"],
                         ["date", "code", "acc_open", "acc_close", "close"])

    def test_missing_proxy_source_is_optional_only_for_independent_raw(self):
        for optional in (True, False):
            with self.subTest(optional=optional):
                self.proxy(optional=optional)
                self.raw["sources"]["market_cap_quotes"] = None
                self.raw["output_dir"] = str(Path(self.temp.name) / f"run_{optional}")
                result = audit(self.load())
                self.assertEqual(result["gate_passed"], optional, result["blocking"])
                if optional:
                    self.assertTrue(any("市值代理来源不可用" in value for value in result["warnings"]))

    def test_adjusted_price_missing_is_still_blocking_in_optional_proxy_mode(self):
        config = self.proxy(shared=True)
        quotes = pd.read_feather(config.source("quotes")).drop(columns="acc_open")
        quotes.to_feather(config.source("quotes"))
        result = audit(config)
        self.assertFalse(result["gate_passed"])
        self.assertTrue(any("acc_open" in value for value in result["blocking"]))

    def test_present_proxy_duplicate_keys_block_even_when_controls_optional(self):
        config = self.proxy()
        path = config.source("market_cap_quotes")
        quotes = pd.read_parquet(path)
        pd.concat([quotes, quotes.iloc[:1]], ignore_index=True).to_parquet(path, index=False)
        result = audit(config)
        self.assertFalse(result["gate_passed"])
        self.assertTrue(any("market_cap_quotes" in value and "duplicate" in value for value in result["blocking"]))

    def test_invalid_units_and_adjusted_price_fields_are_rejected(self):
        self.proxy()
        original = copy.deepcopy(self.raw)
        cases = [("volume_unit", "shares"), ("turnover_unit", "fraction"),
                 ("price_basis", "adjusted"), ("close_column", "acc_close"),
                 ("close_column", "acc_open"), ("method", "barra_size")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.raw = copy.deepcopy(original)
                self.raw["neutralization"]["market_cap"][key] = value
                with self.assertRaises(ValueError):
                    self.load()

    def test_proxy_source_aliases_must_be_declared_but_null_is_allowed(self):
        self.proxy()
        self.raw["sources"].pop("market_cap_quotes")
        with self.assertRaisesRegex(ValueError, "sources"):
            self.load()
        self.raw["sources"]["market_cap_quotes"] = None
        self.assertIsNone(self.load().source("market_cap_quotes"))

    def test_proxy_numeric_columns_cannot_alias_keys_or_each_other(self):
        self.proxy(shared=True)
        original = copy.deepcopy(self.raw)
        cases = [("close_column", "volume"), ("turnover_column", "close"),
                 ("turnover_column", "volume"), ("close_column", "date"),
                 ("volume_column", "code"), ("turnover_column", "date")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.raw = copy.deepcopy(original)
                self.raw["neutralization"]["market_cap"][key] = value
                with self.assertRaises(ValueError):
                    self.load()


if __name__ == "__main__":
    unittest.main()
