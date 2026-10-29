"""Check browser calculations against the backend using the shipped app.js.

Node is optional for Python-only installations. Set REPORT_TEST_NODE to its
executable path when it is not on PATH. DOM/chart stubs only capture displayed
results; statistical functions and their numeric helpers come from app.js.
"""
from datetime import date, timedelta
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from fundamental_factor.backtest import performance_metrics
from fundamental_factor.inference import mean_interval


class ReportFrontendMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = os.environ.get("REPORT_TEST_NODE") or shutil.which("node")
        if not cls.node:
            raise unittest.SkipTest("Node unavailable; set REPORT_TEST_NODE to run report frontend math checks")
        app = (Path(__file__).resolve().parents[1] /
               "fundamental_factor/report_assets/app.js").read_text(encoding="utf-8")
        # Declarations are flush-left; statements within these functions are
        # indented. Stop at the next declaration so no renderer is initialized.
        functions = []
        for name in ("inRange", "pRows", "icRows", "benchmark", "table",
                     "performance", "perf", "icStats", "icAnalysis", "annual"):
            found = re.search(r"^function " + name + r"\b.*?(?=^(?:function|const|let) |\Z)",
                              app, re.M | re.S)
            if not found:
                raise AssertionError(f"Could not extract report function {name}")
            functions.append(found.group(0))
        helpers = []
        for name in ("$", "mean", "sd", "fmt", "signed", "esc", "date"):
            found = re.search(r"^const " + re.escape(name) + r"=.*$", app, re.M)
            if not found:
                raise AssertionError(f"Could not extract report helper {name}")
            helpers.append(found.group(0))
        cls.source = "\n".join(helpers + functions)
        cls.dates = [(date(2024, 1, 1) + timedelta(weeks=i)).isoformat() for i in range(31)]

    def browser(self, body, *, portfolio=None, ic=None):
        fixtures = {"dates": self.dates, "portfolio": portfolio or [], "ic": ic or []}
        script = """
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const elements={},captures={};
const document={getElementById:id=>elements[id]||(elements[id]={
 textContent:'',innerHTML:'',querySelectorAll:()=>[]})};
const ANN=52,ROLL=52,PERIOD='周',CFG={metrics:{nw_lag:13}},START='';
const S={range:'3',ic:'spearman',variant:'raw',trend:'cumulative',annual:'Q10'};
const F={portfolio:input.portfolio,ic:input.ic},D={benchmark:input.portfolio},DATES=input.dates;
function lineChart(id,series){captures[id]=series}
function color(){return ''}
""" + self.source + "\nconsole.log(JSON.stringify((()=>{\n" + body + "\n})()));"
        result = subprocess.run([self.node, "-e", script], input=json.dumps(fixtures),
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def portfolio(self, returns, excess=None):
        rows = [{"entry": self.dates[i], "exit": self.dates[i + 1],
                 "variant": "raw", "portfolio": "Q10", "return": value}
                for i, value in enumerate(returns)]
        for row, value in zip(rows, excess or []):
            row["excess_return"] = value
        return rows

    def ic_rows(self):
        return [{"entry": self.dates[i], "variant": "raw", "method": "spearman",
                 "subset": "ALL", "ic": (i % 7 - 3) * .01 + i * .002}
                for i in range(30)]

    def test_newey_west_interval_matches_backend_finite_sample_correction(self):
        rows = self.ic_rows()
        reference = mean_interval([r["ic"] for r in rows], 13)
        actual = self.browser("return icStats(icRows());", ic=rows)
        self.assertEqual(actual["n"], reference["n"])
        self.assertAlmostEqual(actual["low"], reference["lower"], places=12)
        self.assertAlmostEqual(actual["high"], reference["upper"], places=12)
        self.assertAlmostEqual(actual["nw"], reference["mean"] / reference["se"], places=12)

    def test_newey_west_is_unavailable_for_null_or_absent_periods(self):
        rows = self.ic_rows()
        rows[5]["ic"] = None
        self.assertIsNone(self.browser("return icStats(icRows());", ic=rows)["nw"])
        del rows[5]
        self.assertIsNone(self.browser("return icStats(icRows());", ic=rows)["nw"])

    def test_newey_west_is_unavailable_for_insufficient_history(self):
        actual = self.browser("return icStats(icRows());", ic=self.ic_rows()[:14])
        self.assertIsNone(actual["nw"])
        self.assertIsNone(actual["low"])
        self.assertIsNone(actual["high"])

    def test_constant_return_sharpe_is_unavailable(self):
        actual = self.browser("return performance(pRows());", portfolio=self.portfolio([.1, .1, .1]))
        self.assertIsNone(actual["sharpe"])

    def test_actual_bankruptcy_stops_at_the_zero_capital_period(self):
        actual = self.browser("return performance(pRows());", portfolio=self.portfolio([.1, -1, .5]))
        self.assertEqual(actual["n"], 2)
        self.assertEqual(actual["total"], -1)
        self.assertEqual(actual["annual"], -1)
        self.assertEqual(actual["dd"], 1)
        self.assertEqual(actual["win"], .5)

    def test_insolvency_stops_without_inventing_a_negative_nav_return(self):
        actual = self.browser("return performance(pRows());", portfolio=self.portfolio([.1, -1.2, .5]))
        self.assertEqual(actual["n"], 2)
        self.assertIsNone(actual["total"])
        self.assertIsNone(actual["annual"])
        self.assertIsNone(actual["dd"])

    def test_cumulative_ic_restarts_after_missing_observations(self):
        rows = self.ic_rows()[:4]
        for row, value in zip(rows, [.1, None, .2, .3]):
            row["ic"] = value
        points = self.browser("icAnalysis(); return captures.lgICTrend[0].points;", ic=rows)
        self.assertEqual([point["value"] for point in points], [.1, None, .2, .5])

    def test_partial_year_label_checks_both_start_and_end(self):
        for first, last, partial in (("02", "12", True), ("01", "11", True), ("01", "12", False)):
            with self.subTest(first=first, last=last):
                rows = self.portfolio([.1, .2], [0, .1])
                rows[0].update(entry=f"2013-{first}-01", exit="2013-03-01")
                rows[1].update(entry=f"2013-{last}-01", exit="2014-01-01")
                table = self.browser("annual(); return elements.lgAnnualTable.innerHTML;", portfolio=rows)
                self.assertEqual("未满年" in table, partial)

    def test_excess_sharpe_uses_all_periods_when_actual_capital_survives(self):
        returns, excess = [.1, .2, .3], [.1, -1.2, .4]
        reference = performance_metrics(returns, excess_returns=excess)
        actual = self.browser("return perf('Q10','raw',true);", portfolio=self.portfolio(returns, excess))
        self.assertEqual(reference["excess_sharpe_n_periods"], 3)
        self.assertAlmostEqual(actual["sharpe"], reference["excess_sharpe"], places=12)

    def test_excess_sharpe_stops_at_actual_bankruptcy(self):
        returns, excess = [.1, -1, .3], [.1, -.9, .4]
        reference = performance_metrics(returns, excess_returns=excess)
        actual = self.browser("return perf('Q10','raw',true);", portfolio=self.portfolio(returns, excess))
        self.assertEqual(reference["excess_sharpe_n_periods"], 2)
        self.assertAlmostEqual(actual["sharpe"], reference["excess_sharpe"], places=12)
