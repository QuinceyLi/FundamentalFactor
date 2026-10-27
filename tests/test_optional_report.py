"""Report terminology must follow the actual frequency and signal contract."""
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


class OptionalReportTests(unittest.TestCase):
    @staticmethod
    def config():
        return {"schema_version": 2, "dataset_kind": "synthetic",
                "factors": [{"name": "ROIC_new", "label": "ROIC"}],
                "backtest": {"rebalance_frequency": "monthly"},
                "preprocess": {"winsorize": False, "standardize": False},
                "neutralization": {"variants": ["raw"], "include_float_mcap": False,
                                   "sample_policy": "independent"},
                "data_policy": {"mode": "classic", "missing_returns": "available_case",
                                "empty_groups": "cash"}}

    def test_monthly_definitions_disclose_direct_values_independent_samples_and_units(self):
        config = self.config()
        body = _definitions(config)
        for expected in ("每月首个交易日", "按该月首个交易日开盘", "年化频数为 12",
                         "配置滞后：3 月", "去极值关闭", "标准化关闭", "signal=value",
                         "raw 只要求有效因子值", "样本差异", "periods.parquet"):
            self.assertIn(expected, body)
        for misleading in ("每周", "该周", "周收益", "周度", "年化频数为 52",
                           "所选信号使用相同股票样本"):
            self.assertNotIn(misleading, body)
        self.assertEqual(_variant_labels(config)["raw"], "原样输入信号")

    def test_enabled_processing_is_not_claimed_to_preserve_input(self):
        config = self.config()
        config["preprocess"] = {"winsorize": True, "standardize": False}
        body = _definitions(config)
        self.assertIn("MAD 截尾", body)
        self.assertIn("标准化关闭", body)
        self.assertIn("raw 表示未中性化版本", body)
        self.assertNotIn("signal=value", body)

    def test_coverage_uses_each_actual_variant_instead_of_intersection(self):
        config = self.config()
        config["neutralization"]["variants"] = ["raw", "industry", "full"]
        coverage = pd.DataFrame({"entry": ["2025-02-05"], "raw_coverage": [.8],
                                 "common_coverage": [.5], "variant_raw_coverage": [.8],
                                 "variant_industry_coverage": [.7], "variant_full_coverage": [.5]})
        series = {item["name"]: item["points"][0]["value"] for item in _coverage_series(coverage, config)}
        self.assertEqual(series["原样输入信号测试覆盖率"], .8)
        self.assertEqual(series["行业中性化信号测试覆盖率"], .7)
        self.assertEqual(series["全中性化信号测试覆盖率"], .5)
        self.assertEqual(series["控制字段交集（仅诊断）"], .5)
        self.assertNotIn("所选信号共同样本覆盖率", series)

    def test_complete_monthly_html_uses_monthly_titles_and_diagnostics(self):
        config = self.config()
        signals, labels, benchmark, weeks = _fixture(first_low=.01, first_high=.1)
        original = pd.DatetimeIndex(sorted(set(weeks.entry) | set(weeks.exit)))
        monthly = pd.to_datetime(["2015-01-05", "2015-02-02", "2015-03-02", "2015-04-01"])
        date_map = dict(zip(original, monthly))
        for frame in (signals, labels, benchmark, weeks):
            for column in ("entry", "exit"):
                frame[column] = frame[column].map(date_map)
        tables = evaluate_factor(signals, labels, benchmark, weeks, annualization=12, nw_lag=3,
                                 missing_returns="available_case", empty_groups="cash")
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "factors" / "ROIC_new"
            folder.mkdir(parents=True)
            metadata = {
                "config.frozen.json": config,
                "audit.json": {"config_digest": digest, "gate_passed": True, "sources": {}},
                "panel_manifest.json": {"config_digest": digest, "status": "ready", "periods": 3,
                                        "unresolved_labels": 0, "incomplete_benchmark_weeks": 0},
                "results.json": {"config_digest": digest, "schema_version": 2,
                                 "dataset_kind": "synthetic", "status": "completed", "factors": [{
                                     "name": "ROIC_new", "output_dir": "factors/ROIC_new",
                                     "sample": {"periods": 3, "weeks": 3, "rebalance_frequency": "monthly"}}]},
            }
            for filename, value in metadata.items():
                (root / filename).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            for name, table in tables.items():
                table.to_csv(folder / f"{name}.csv", index=False)
            pd.DataFrame({"factor": ["ROIC_new"], "entry": ["2015-01-05"],
                          "raw_coverage": [1.], "common_coverage": [.5],
                          "variant_raw_coverage": [1.]}).to_csv(root / "coverage.csv", index=False)
            report(root)
            body = (folder / "report.html").read_text()
            for expected in ("月度 RankIC", "原样输入信号", "逐月因子覆盖与各版本测试样本",
                             "基准收益无法计算月数", "年化频数为 12", "日历月数", "调仓频率", "持有期数"):
                self.assertIn(expected, body)
            for misleading in ("所选信号共享样本", "共同样本 ALL", "周度 RankIC", "日历周数"):
                self.assertNotIn(misleading, body)
            self.assertNotIn("<script src=", body)


if __name__ == "__main__":
    unittest.main()
