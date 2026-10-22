"""Classic reports separate usable factor results from optional data gaps."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fundamental_factor.backtest import evaluate_factor
from fundamental_factor.report import report
from tests.test_backtest_edges import _fixture


class ClassicReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "factors" / "ROIC_new"
        self.folder.mkdir(parents=True)
        self.config = {"schema_version": 2, "dataset_kind": "synthetic",
                       "factors": [{"name": "ROIC_new", "label": "ROIC"}],
                       "neutralization": {"variants": ["raw"], "include_float_mcap": False},
                       "data_policy": {"mode": "classic"},
                       "execution": {"enabled": True}}
        self.digest = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self.audit = {"config_digest": self.digest, "gate_passed": True, "sources": {},
                      "warnings": ["上海历史 ST 未覆盖，保持未知。"],
                      "execution_gate_passed": False,
                      "execution_blocking": ["缺少原始股数账本公司行动覆盖确认。"]}
        self._json("config.frozen.json", self.config)
        self._json("audit.json", self.audit)
        self._json("panel_manifest.json", {"config_digest": self.digest, "status": "ready",
                   "quality_status": "unresolved_labels", "base_rows": 30,
                   "unknown_st_base_rows": 12, "unresolved_labels": 1,
                   "label_coverage": 29 / 30, "incomplete_benchmark_weeks": 1})
        self._json("results.json", {"config_digest": self.digest, "schema_version": 2,
                   "dataset_kind": "synthetic", "status": "completed", "factors": [{
                   "name": "ROIC_new", "output_dir": "factors/ROIC_new", "execution": {
                       "requested": True, "enabled": False, "status": "skipped",
                       "blocking": self.audit["execution_blocking"]}}]})
        signals, labels, benchmark, weeks = _fixture(first_low=.01, first_high=.1)
        labels.loc[0, ["return", "unresolved"]] = [np.nan, True]
        benchmark.loc[0, "return"] = np.nan
        tables = evaluate_factor(signals, labels, benchmark, weeks, missing_returns="report")
        for name, table in tables.items():
            table.to_csv(self.folder / f"{name}.csv", index=False)

    def _json(self, name, value):
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _render(self):
        report(self.root)
        return ((self.root / "report.html").read_text(),
                (self.folder / "report.html").read_text())

    def test_incomplete_optional_data_keeps_current_factor_results_visible(self):
        index, factor = self._render()
        for text in (index, factor):
            self.assertIn("经典因子统计口径", text)
            self.assertIn("计算状态与数据完整性", text)
            self.assertIn("40.00%", text)
            self.assertIn("96.66667%", text)
            self.assertIn("未解决收益股周", text)
            self.assertNotIn("尚无正式研究结论", text)
        self.assertIn("当前单因子回测已完成", index)
        self.assertIn("RankIC 回测（主指标）", factor)
        self.assertIn("十分组组合表现", factor)
        self.assertIn("缺失持仓标签使该组合当周收益缺失", factor)

    def test_raw_only_report_does_not_claim_neutralization_or_barra_requirements(self):
        index, factor = self._render()
        for text in (index, factor):
            self.assertIn("本次只测试原始信号，不要求行业、市值或 Barra", text)
            self.assertNotIn("全中性使用的风险暴露", text)
            self.assertNotIn("三套信号", text)
        self.assertNotIn('id="full"', factor)
        self.assertNotIn('id="industry"', factor)

    def test_optional_execution_failure_is_explained_without_hiding_theory(self):
        _, factor = self._render()
        self.assertIn("可选执行账本已跳过，因子统计已独立完成", factor)
        self.assertIn("缺少原始股数账本公司行动覆盖确认", factor)
        self.assertNotIn('href="execution_', factor)
        self.assertIn("十分组组合表现", factor)

    def test_core_integrity_failure_still_hides_successful_old_tables(self):
        self.audit.update(gate_passed=False, blocking=["核心价格出现重复键。"])
        self._json("audit.json", self.audit)
        index, factor = self._render()
        for text in (index, factor):
            self.assertIn("尚无正式研究结论", text)
            self.assertIn("核心价格出现重复键", text)
        self.assertNotIn("RankIC 回测（主指标）", factor)


if __name__ == "__main__":
    unittest.main()
