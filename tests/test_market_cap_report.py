"""Market-cap reports must distinguish a quote proxy from verified market cap."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fundamental_factor.backtest import evaluate_factor
from fundamental_factor.report import _coverage_series, _definitions, _variant_labels, report
from tests.test_backtest_edges import _fixture


class MarketCapReportTests(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "schema_version": 2, "dataset_kind": "synthetic",
            "factors": [{"name": "ROIC_new", "label": "ROIC"}],
            "preprocess": {"winsorize": False, "standardize": False},
            "neutralization": {
                "variants": ["raw", "size"], "sample_policy": "independent",
                "include_float_mcap": True,
                "market_cap": {
                    "method": "quote_proxy", "quotes_source": "market_cap_quotes",
                    "turnover_source": "quotes", "close_column": "close",
                    "volume_column": "volume", "turnover_column": "turnover",
                    "volume_unit": "lot", "turnover_unit": "percent", "price_basis": "unadjusted",
                },
            },
            "data_policy": {"mode": "classic", "missing_returns": "available_case", "empty_groups": "cash"},
        }

    def test_size_definitions_disclose_proxy_timing_units_and_independent_inputs(self):
        body = _definitions(self.config())
        for expected in (
            "纯市值中性仅使用截距和行情代理市值的自然对数，不要求行业或 Barra",
            "close * volume * 100 / (turnover / 100)", "未复权收盘价", "1 手=100 股",
            "1 表示 1%", "该代理不是已核验的实际流通市值", "同一天 date/code 连接",
            "调仓截点前最近一条记录", "禁止使用调仓当日完整成交信息",
            "market_cap_date", "最新记录无效时不越过它", "log(market_cap_proxy)",
            "代理缺失只影响需要市值的版本", "raw 只要求有效因子值，不受行业、市值或 Barra",
        ):
            self.assertIn(expected, body)
        self.assertNotIn("行业市值中性使用", body)
        self.assertNotIn("全中性使用", body)

    def test_history_method_remains_actual_cap_and_legacy_variants_do_not_change(self):
        config = self.config()
        del config["neutralization"]["market_cap"]
        body = _definitions(config)
        self.assertIn("纯市值中性仅使用截距和个股实际流通市值的自然对数", body)
        self.assertIn("市值来源为 history.float_mcap，要求实际流通市值", body)
        self.assertNotIn("市值代理公式", body)
        self.assertEqual(list(_variant_labels({})), ["raw", "industry_size", "full"])
        self.assertEqual(_variant_labels({}, values=["size"]), {"size": "市值中性化信号"})

    def test_proxy_configuration_does_not_claim_cap_requirement_when_unused(self):
        config = self.config()
        config["neutralization"]["variants"] = ["raw"]
        body = _definitions(config)
        self.assertIn("本次只测试原始信号，不要求行业、市值或 Barra", body)
        self.assertNotIn("市值代理公式", body)
        config["neutralization"]["variants"] = ["raw", "industry", "full"]
        config["neutralization"]["include_float_mcap"] = False
        body = _definitions(config)
        self.assertIn("纯行业中性仅使用行业标签", body)
        self.assertIn("全中性使用行业及冻结配置", body)
        self.assertNotIn("市值代理公式", body)

    def test_industry_size_and_full_disclose_selected_cap_source(self):
        config = self.config()
        config["neutralization"]["variants"] = ["industry_size", "full"]
        body = _definitions(config)
        self.assertIn("行业市值中性使用行业和行情代理市值的自然对数", body)
        self.assertIn("全中性使用行业、行情代理市值的自然对数", body)
        self.assertIn("市值代理公式", body)
        self.assertNotIn("个股实际流通市值的自然对数", body)

    def test_coverage_preserves_distinct_raw_and_size_samples(self):
        frame = pd.DataFrame({"entry": ["2025-02-05"], "raw_coverage": [1.],
                              "common_coverage": [.7], "variant_raw_coverage": [1.],
                              "variant_size_coverage": [.7]})
        series = {item["name"]: item["points"][0]["value"]
                  for item in _coverage_series(frame, self.config())}
        self.assertEqual(series["原样输入信号测试覆盖率"], 1.)
        self.assertEqual(series["市值中性化信号测试覆盖率"], .7)
        self.assertEqual(series["控制字段交集（仅诊断）"], .7)

    def test_complete_report_renders_size_results_and_proxy_contract(self):
        config = self.config()
        signals, labels, benchmark, weeks = _fixture(first_low=.01, first_high=.1)
        signals = pd.concat([signals, signals.assign(variant="size")], ignore_index=True)
        tables = evaluate_factor(signals, labels, benchmark, weeks,
                                 missing_returns="available_case", empty_groups="cash")
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "factors" / "ROIC_new"
            folder.mkdir(parents=True)
            metadata = {
                "config.frozen.json": config,
                "audit.json": {"config_digest": digest, "gate_passed": True, "sources": {}},
                "panel_manifest.json": {"config_digest": digest, "status": "ready", "periods": 3},
                "results.json": {"config_digest": digest, "schema_version": 2,
                                 "dataset_kind": "synthetic", "status": "completed", "factors": [{
                                     "name": "ROIC_new", "output_dir": "factors/ROIC_new",
                                     "sample": {"periods": 3, "weeks": 3, "rebalance_frequency": "weekly"}}]},
            }
            for filename, value in metadata.items():
                (root / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            for name, table in tables.items():
                table.to_csv(folder / f"{name}.csv", index=False)
            pd.DataFrame({"factor": ["ROIC_new"], "entry": ["2020-01-06"],
                          "raw_coverage": [1.], "common_coverage": [1.],
                          "variant_raw_coverage": [1.], "variant_size_coverage": [1.]}).to_csv(
                              root / "coverage.csv", index=False)
            report(root)
            body = (folder / "report.html").read_text(encoding="utf-8")
            for expected in ("市值中性化信号", "市值中性化信号测试覆盖率", "市值代理公式", "log(market_cap_proxy)"):
                self.assertIn(expected, body)
            self.assertNotIn("<script src=", body)


if __name__ == "__main__":
    unittest.main()
