"""Classic analytics require factor/price integrity, not complete execution records."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fundamental_factor.audit import audit, source_columns
from fundamental_factor.config import load_config
from fundamental_factor.panel import build_panel
from tests.fixtures import disk_fixture


def classic_fixture(root, variants=None):
    config = disk_fixture(root, weeks_count=3)
    raw = json.loads(config.path.read_text())
    raw.pop("data_policy")
    if variants is not None:
        raw["neutralization"] = {"include_float_mcap": False, "variants": variants}
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


def rewrite(config, **changes):
    raw = json.loads(config.path.read_text())
    raw.update(changes)
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


class ClassicAuditTests(unittest.TestCase):
    def test_default_classic_and_explicit_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp))
            self.assertEqual(config.data_policy["mode"], "classic")
            self.assertEqual(config.data_policy["st_unknown"], "include")
            self.assertEqual(config.data_policy["suspension_unknown"], "include")
            self.assertEqual(config.data_policy["missing_returns"], "report")
            config = rewrite(config, data_policy={"mode": "strict"})
            self.assertEqual(config.data_policy["missing_returns"], "error")
            config = rewrite(config, data_policy={"mode": "proxy", "description": "Explicit legacy proxy."})
            self.assertEqual(config.data_policy["st_unknown"], "error")

    def test_raw_accepts_absent_optional_sources_and_unverified_declarations(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp), ["raw"])
            sources = config.raw["sources"].copy()
            for name in ("history", "status", "risk", "terminal"):
                sources[name] = None
            config = rewrite(config, sources=sources, point_in_time_confirmed=False,
                             point_in_time_evidence="", price_contract={})
            checked = audit(config)
            self.assertTrue(checked["gate_passed"], checked["blocking"])
            self.assertNotIn("risk", source_columns(config))
            self.assertGreaterEqual(len(checked["warnings"]), 4)
            self.assertFalse(config.requires_industry)
            self.assertFalse(config.requires_float_mcap)

    def test_industry_requires_category_but_not_st_risk_or_market_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp), ["raw", "industry"])
            history = pd.read_parquet(config.source("history")).drop(columns=["is_st", "float_mcap"])
            history.to_parquet(config.source("history"), index=False)
            sources = config.raw["sources"].copy()
            sources.update(risk=None, status=None)
            config = rewrite(config, sources=sources)
            self.assertEqual(source_columns(config)["history"], ["date", "code", "industry"])
            checked = audit(config)
            self.assertTrue(checked["gate_passed"], checked["blocking"])
            self.assertEqual(checked["sources"]["history"]["unknown_flags"]["is_st"], len(history))

    def test_requested_control_source_still_required(self):
        for variant, missing in [("industry", "history"), ("full", "risk")]:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                config = classic_fixture(Path(tmp), [variant])
                sources = config.raw["sources"].copy()
                sources[missing] = None
                checked = audit(rewrite(config, sources=sources))
                self.assertFalse(checked["gate_passed"])
                self.assertTrue(any(missing in error for error in checked["blocking"]))

    def test_empty_optional_sources_reach_panel_and_preserve_unknown_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp), ["raw"])
            for name in ("history", "status"):
                pd.read_parquet(config.source(name)).iloc[:0].to_parquet(config.source(name), index=False)
            terminal_path = Path(config.raw["data_root"]) / "terminal.csv"
            pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                          "available_date": pd.Series(dtype="datetime64[ns]"),
                          "code": pd.Series(dtype="string"),
                          "terminal_value": pd.Series(dtype="float64")}).to_csv(terminal_path, index=False)
            stocks = pd.read_parquet(config.source("stocks"))
            stocks.loc[stocks.code.eq("000003.SZ"), "delisting_date"] = pd.Timestamp("2015-01-09")
            stocks.to_parquet(config.source("stocks"), index=False)
            quotes = pd.read_feather(config.source("quotes"))
            quotes = quotes[~(quotes.code.eq("000003.SZ") & quotes.date.ge("2015-01-09"))]
            quotes.reset_index(drop=True).to_feather(config.source("quotes"))
            sources = config.raw["sources"].copy()
            sources["terminal"] = str(terminal_path)
            config = rewrite(config, sources=sources)
            manifest = build_panel(config)
            self.assertEqual(manifest["status"], "ready")
            checked = json.loads((config.output / "audit.json").read_text())
            for name in ("history", "status", "terminal"):
                self.assertTrue(checked["sources"][name]["empty_optional"])
                self.assertTrue(any(f"可选 {name} 数据为空" in text for text in checked["warnings"]))
            universe = pd.read_parquet(config.output / "universe.parquet")
            self.assertTrue(universe.is_st_unknown.all())
            self.assertTrue(universe.entry_suspension_unknown.all())
            labels = pd.read_parquet(config.output / "labels.parquet")
            self.assertEqual(int(labels.unresolved.sum()), 1)
            self.assertTrue(labels.loc[labels.unresolved, "return"].isna().all())
            self.assertTrue(labels.loc[~labels.unresolved, "return"].notna().all())
            self.assertEqual(labels.loc[labels.unresolved, "code"].tolist(), ["000003.SZ"])
            self.assertTrue(labels.trade_restriction_unknown.all())
            self.assertFalse(labels.terminal_applied.any())

    def test_empty_core_or_selected_control_sources_remain_errors(self):
        cases = [("quotes", ["raw"]), ("roic", ["raw"]),
                 ("history", ["industry"]), ("risk", ["full"])]
        for name, variants in cases:
            with self.subTest(source=name), tempfile.TemporaryDirectory() as tmp:
                config = classic_fixture(Path(tmp), variants)
                path = config.source(name)
                if path.suffix == ".feather":
                    pd.read_feather(path).iloc[:0].to_feather(path)
                else:
                    pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
                checked = audit(config)
                self.assertFalse(checked["gate_passed"])
                self.assertTrue(any(f"{name} 接入校验失败：empty source" in error
                                    for error in checked["blocking"]))

    def test_status_can_supply_limits_without_suspension_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp), ["raw"])
            status = pd.read_parquet(config.source("status")).drop(columns="suspended")
            status.to_parquet(config.source("status"), index=False)
            self.assertEqual(source_columns(config)["status"], ["date", "code"])
            checked = audit(config)
            self.assertTrue(checked["gate_passed"], checked["blocking"])
            self.assertEqual(checked["sources"]["status"]["unknown_flags"]["suspended"], len(status))

    def test_unreadable_optional_source_is_reported_in_audit_not_requirements_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp), ["raw"])
            config.source("history").write_text("invalid parquet bytes")
            checked = audit(config)
            self.assertFalse(checked["gate_passed"])
            self.assertTrue(any("history 接入校验失败" in error for error in checked["blocking"]))

    def test_incomplete_raw_ledger_skips_only_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp), ["raw"])
            config = rewrite(config, execution={"enabled": True, "price_mode": "raw"})
            checked = audit(config)
            self.assertTrue(checked["gate_passed"], checked["blocking"])
            self.assertFalse(checked["execution_gate_passed"])
            self.assertTrue(any("公司行动" in error for error in checked["execution_blocking"]))

    def test_factor_price_duplicate_keys_are_not_relaxed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp), ["raw"])
            quotes = pd.read_feather(config.source("quotes"))
            pd.concat([quotes, quotes.iloc[:1]], ignore_index=True).to_feather(config.source("quotes"))
            checked = audit(config)
            self.assertFalse(checked["gate_passed"])
            self.assertTrue(any("quotes" in error for error in checked["blocking"]))

    def test_adjusted_units_accepts_default_classic_but_double_adjustment_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_fixture(Path(tmp), ["raw"])
            sources = config.raw["sources"].copy()
            sources["execution_status"] = sources["status"]
            sources["corporate_actions"] = "/nonexistent/actions.csv"
            config = rewrite(config, sources=sources, execution={"enabled": True,
                "price_mode": "adjusted_units", "description": "Adjusted units only."})
            checked = audit(config)
            self.assertTrue(checked["gate_passed"])
            self.assertFalse(checked["execution_gate_passed"])
            self.assertTrue(any("重复调整" in error for error in checked["execution_blocking"]))


if __name__ == "__main__":
    unittest.main()
