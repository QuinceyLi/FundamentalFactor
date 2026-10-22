"""Acceptance checks for optional execution accounts in the offline report."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fundamental_factor.backtest import evaluate_factor, performance_metrics
from fundamental_factor.report import report


class ExecutionReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="factor-execution-report-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "factors" / "ROIC_new"
        self.folder.mkdir(parents=True)
        self.config = {"schema_version": 2, "dataset_kind": "synthetic",
                       "factors": [{"name": "ROIC_new", "label": "ROIC"}],
                       "metrics": {"annualization": 52, "nw_lag": 13}}
        digest = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        self._json("config.frozen.json", self.config)
        self._json("audit.json", {"config_digest": digest, "gate_passed": True, "sources": {}})
        self._json("panel_manifest.json", {"config_digest": digest, "status": "ready"})
        self.result = {"schema_version": 2, "config_digest": digest, "dataset_kind": "synthetic",
                       "status": "completed", "factors": [{"name": "ROIC_new", "label": "ROIC",
                       "output_dir": "factors/ROIC_new", "execution": {"enabled": True}}]}
        self._json("results.json", self.result)
        entries = pd.date_range("2020-01-06", periods=3, freq="7D")
        self.weeks = pd.DataFrame({"entry": entries[:-1], "exit": entries[1:]})
        signals, labels = [], []
        for week in self.weeks.itertuples(index=False):
            for i in range(10):
                base = {"entry": week.entry, "exit": week.exit, "code": f"{i:06d}.SZ"}
                signals.append(base | {"variant": "raw", "signal": float(i), "percentile": (i + 1) / 10, "group": i + 1})
                labels.append(base | {"return": .001 * i, "unresolved": False, "trade_restricted": False})
        tables = evaluate_factor(pd.DataFrame(signals), pd.DataFrame(labels),
                                 self.weeks.assign(**{"return": .002}), self.weeks)
        for name, table in tables.items():
            table.to_csv(self.folder / f"{name}.csv", index=False)
        self._execution_files()

    def _json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def _execution_files(self):
        rows, summaries = [], []
        for portfolio, values in [("Q1", [.1, -.1]), ("Q10", [.2, .05])]:
            nav = 1_000_000.
            for i, week in enumerate(self.weeks.itertuples(index=False)):
                start = nav
                nav *= 1 + values[i]
                rows.append({"entry": week.entry, "exit": week.exit, "variant": "raw", "portfolio": portfolio,
                             "nav_start": start, "return": values[i], "nav": nav, "turnover": 1. if not i else .2,
                             "cash_weight": .2 if not i else .1, "n_stocks": 2, "blocked_buy_value": 50_000. if not i else 0.,
                             "blocked_sell_value": 0., "carried_valuation_count": i, "target_empty": False, "signal_missing": bool(i)})
            summaries.append({"variant": "raw", "portfolio": portfolio,
                              **performance_metrics(values, [1., .2])})
        pd.DataFrame(rows).to_csv(self.folder / "execution_weekly.csv", index=False)
        pd.DataFrame(summaries).to_csv(self.folder / "execution_performance.csv", index=False)
        for kind in ["holdings", "trades", "daily"]:
            pd.DataFrame({"note": ["synthetic artifact"]}).to_parquet(self.folder / f"execution_{kind}.parquet", index=False)

    def _render(self):
        report(self.root)
        return (self.folder / "report.html").read_text(encoding="utf-8")

    def test_enabled_accounts_are_normalized_and_fully_offline(self):
        text = self._render()
        self.assertIn("原始信号：执行账户净值", text)
        self.assertIn("原始信号：执行后现金权重", text)
        self.assertIn("合成数据验收示例", text)
        self.assertIn("现金与持股逐日记账的费前执行模拟", text)
        self.assertIn("缺失信号冻结周数", text)
        self.assertIn("累计沿用估值次数", text)
        self.assertIn("不每日追单", text)
        self.assertIn("各自独立的多头账户", text)
        for suffix in ["weekly.csv", "performance.csv", "holdings.parquet", "trades.parquet", "daily.parquet"]:
            self.assertIn(f'href="execution_{suffix}"', text)
        payloads = [json.loads(value) for value in re.findall(r'<script type="application/json" class="chart-data">(.*?)</script>', text)]
        # Execution charts are appended after the unchanged theoretical charts.
        nav, cash = payloads[-2:]
        by_name = {series["name"]: [point["v"] for point in series["points"]] for series in nav["series"]}
        self.assertEqual(by_name["Q1"][0], 1.)
        self.assertAlmostEqual(by_name["Q1"][-1], .99)
        self.assertAlmostEqual(by_name["Q10"][-1], 1.26)
        self.assertEqual(cash["unit"], "percent")
        self.assertNotIn('<script src=', text)
        self.assertNotIn('<link', text)
        self.assertIn("Q10−Q1 理论多空组合净值", text)

    def test_disabled_card_hides_leftover_execution_values_and_links(self):
        self.result["factors"][0]["execution"]["enabled"] = False
        self._json("results.json", self.result)
        text = self._render()
        self.assertIn("本次未启用执行账本", text)
        self.assertIn("可选复权单位代理账本", text)
        self.assertNotIn("原始信号：执行账户净值", text)
        self.assertNotIn('href="execution_', text)
        self.assertIn("十分组组合表现", text)

    def test_missing_execution_table_keeps_theoretical_results(self):
        (self.folder / "execution_performance.csv").unlink()
        text = self._render()
        self.assertIn("执行账本产物尚不完整", text)
        self.assertNotIn("原始信号：执行账户净值", text)
        self.assertNotIn('href="execution_', text)
        self.assertIn("十分组组合表现", text)

    def test_failure_gate_hides_previous_execution_results(self):
        self._render()
        self._json("failure.json", {"message": "execution data failed"})
        text = self._render()
        self.assertIn("尚无正式研究结论", text)
        self.assertNotIn("原始信号：执行账户净值", text)
        self.assertNotIn('href="execution_', text)

    def test_duplicate_account_weeks_are_rejected(self):
        path = self.folder / "execution_weekly.csv"
        weekly = pd.read_csv(path)
        pd.concat([weekly, weekly.iloc[:1]], ignore_index=True).to_csv(path, index=False)
        text = self._render()
        self.assertIn("执行账本产物未通过核验", text)
        self.assertNotIn("原始信号：执行账户净值", text)

    def test_invalid_first_capital_is_not_replaced_by_a_later_value(self):
        path = self.folder / "execution_weekly.csv"
        weekly = pd.read_csv(path)
        weekly.loc[0, "nav_start"] = float("nan")
        weekly.to_csv(path, index=False)
        text = self._render()
        self.assertIn("账户初始资产无效", text)
        self.assertNotIn("原始信号：执行账户净值", text)


if __name__ == "__main__":
    unittest.main()
