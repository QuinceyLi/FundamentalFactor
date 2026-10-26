"""The report must distinguish computable statistics from complete holdings data."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_factor.backtest import evaluate_factor
from fundamental_factor.report import _portfolio_diagnostics, report
from tests.test_backtest_edges import _fixture


class AvailableReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "factors" / "ROIC_new"
        self.folder.mkdir(parents=True)
        self.config = {"schema_version": 2, "dataset_kind": "synthetic",
                       "factors": [{"name": "ROIC_new", "label": "ROIC"}],
                       "neutralization": {"variants": ["raw"], "include_float_mcap": False},
                       "data_policy": {"mode": "classic", "missing_returns": "available_case",
                                       "empty_groups": "cash"}}
        digest = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self._json("config.frozen.json", self.config)
        self._json("audit.json", {"config_digest": digest, "gate_passed": True, "sources": {}})
        self._json("panel_manifest.json", {"config_digest": digest, "status": "ready",
                   "quality_status": "unresolved_labels", "base_rows": 60,
                   "unresolved_labels": 1, "label_coverage": 59 / 60,
                   "incomplete_benchmark_weeks": 0, "partial_benchmark_weeks": 1})
        self._json("results.json", {"config_digest": digest, "schema_version": 2,
                   "dataset_kind": "synthetic", "status": "completed", "factors": [{
                   "name": "ROIC_new", "output_dir": "factors/ROIC_new"}]})
        signals, labels, benchmark, weeks = _fixture(first_low=.01, first_high=.1)
        # Two stocks per original group. Q10 is truly empty from frozen signals;
        # Q1 has a missing first-week label but retains one observed return.
        second_signals, second_labels = signals.copy(), labels.copy()
        second_signals["code"] = "1" + second_signals.code.str[1:]
        second_labels["code"] = "1" + second_labels.code.str[1:]
        signals = pd.concat([signals, second_signals], ignore_index=True)
        labels = pd.concat([labels, second_labels], ignore_index=True)
        signals.loc[signals.group.eq(10), "group"] = 9
        labels.loc[0, ["return", "unresolved"]] = [np.nan, True]
        self.tables = evaluate_factor(signals, labels, benchmark, weeks,
                                      missing_returns="available_case", empty_groups="cash")
        for name, table in self.tables.items():
            table.to_csv(self.folder / f"{name}.csv", index=False)

    def _json(self, name, value):
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _render(self):
        report(self.root)
        return (self.folder / "report.html").read_text(encoding="utf-8")

    def test_report_discloses_available_weights_without_claiming_complete_labels(self):
        body = self._render()
        for expected in ["统计权重与冻结持仓权重不同", "选择偏差", "target_weight", "evaluation_weight",
                         "非空组全无有效收益", "整周信号缺失", "目标双边换手", "漂移换手缺失仍保持 NA",
                         "平均缺失标签目标权重", "年化目标换手率（双边）",
                         "基准收益无法计算周数", "基准标签不完整但收益可计算周数"]:
            self.assertIn(expected, body)
        self.assertNotIn("任何缺失持仓标签使该组合当周收益缺失", body)
        self.assertNotIn("也不删除股票后重新分配组内权重", body)
        self.assertIn("https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py", body)
        self.assertIn("空组现金是本项目配置约定", body)
        self.assertNotIn("尚无正式研究结论", body)

    def test_chart_uses_computed_continuous_nav_and_cash_without_plot_interpolation(self):
        body = self._render()
        charts = [json.loads(value) for value in re.findall(
            r'<script type="application/json" class="chart-data">(.*?)</script>', body)]
        chart = next(item for item in charts if len(item["series"]) == 10)
        points = {series["name"]: series["points"] for series in chart["series"]}
        q1_values = [point["v"] for point in points["Q1"]]
        self.assertEqual(q1_values[0], 1.)
        self.assertEqual(len(q1_values), 4)
        self.assertAlmostEqual(q1_values[-1], 1.01 * 1.5 * 1.5)
        self.assertEqual([point["v"] for point in points["Q10"]], [1., 1., 1., 1.])
        self.assertNotIn('<script src=', body)

    def test_diagnostics_preserve_partial_label_weight_even_when_all_returns_compute(self):
        diagnostics = _portfolio_diagnostics(self.tables["portfolio_weekly"]).set_index("portfolio")
        self.assertEqual(diagnostics.loc["Q1", "valid_return_weeks"], 3)
        self.assertEqual(diagnostics.loc["Q1", "incomplete_label_weeks"], 1)
        self.assertAlmostEqual(diagnostics.loc["Q1", "max_missing_weight"], .5)
        self.assertEqual(diagnostics.loc["Q10", "empty_weeks"], 3)
        self.assertEqual(diagnostics.loc["Q10", "missing_return_weeks"], 0)
        self.assertEqual(diagnostics.loc["Q10", "mean_cash_weight"], 1.)


if __name__ == "__main__":
    unittest.main()
