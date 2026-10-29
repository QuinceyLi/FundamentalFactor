"""Acceptance checks for optional execution accounts in the offline report."""
from __future__ import annotations

import hashlib
import json
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
        return json.loads((self.root / "report_data.json").read_text())

    def test_enabled_accounts_are_preserved_and_report_is_offline(self):
        data = self._render()
        account = data["factors"][0]["execution"]
        self.assertFalse(account["issues"])
        rows = pd.DataFrame(account["weekly"])
        q1 = rows[rows.portfolio.eq("Q1")]
        q10 = rows[rows.portfolio.eq("Q10")]
        self.assertAlmostEqual(q1.nav.iloc[-1] / q1.nav_start.iloc[0], .99)
        self.assertAlmostEqual(q10.nav.iloc[-1] / q10.nav_start.iloc[0], 1.26)
        self.assertEqual(q1.cash_weight.iloc[0], .2)
        self.assertEqual(data["meta"]["dataset_kind"], "synthetic")
        self.assertNotIn('<script src=', (self.root / 'report.html').read_text())

    def test_disabled_status_hides_leftover_execution_values(self):
        self.result["factors"][0]["execution"]["enabled"] = False
        self._json("results.json", self.result)
        data = self._render()
        self.assertFalse(data["factors"][0]["execution"]["weekly"])
        self.assertTrue(data["factors"][0]["portfolio"])

    def test_missing_execution_table_keeps_theory_and_explains_execution_gap(self):
        (self.folder / "execution_performance.csv").unlink()
        data = self._render()
        self.assertTrue(data["factors"][0]["execution"]["issues"])
        self.assertFalse(data["factors"][0]["execution"]["weekly"])
        self.assertTrue(data["factors"][0]["portfolio"])

    def test_failure_gate_hides_previous_execution_results(self):
        self._render()
        self._json("failure.json", {"message": "execution data failed"})
        data = self._render()
        self.assertEqual(data["factors"], [])
        self.assertTrue(data["issues"])

    def test_duplicate_account_periods_are_rejected(self):
        path = self.folder / "execution_weekly.csv"
        weekly = pd.read_csv(path)
        pd.concat([weekly, weekly.iloc[:1]], ignore_index=True).to_csv(path, index=False)
        data = self._render()
        self.assertTrue(data["factors"][0]["execution"]["issues"])
        self.assertFalse(data["factors"][0]["execution"]["weekly"])

    def test_invalid_first_capital_is_not_replaced_by_later_value(self):
        path = self.folder / "execution_weekly.csv"
        weekly = pd.read_csv(path)
        weekly.loc[0, "nav_start"] = float("nan")
        weekly.to_csv(path, index=False)
        data = self._render()
        self.assertTrue(data["factors"][0]["execution"]["issues"])
        self.assertFalse(data["factors"][0]["execution"]["weekly"])


if __name__ == "__main__":
    unittest.main()
