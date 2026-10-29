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
        for name in ("inRange", "signalEntries", "pRows", "icRows", "benchmark", "table",
                     "compound", "performance", "perf", "icStats", "rolling", "cumulativeIC",
                     "icAnalysis", "annual", "empty", "recordsTable", "evaluationDisclosure", "refresh"):
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
        cls.chart_source = re.search(
            r"^function lineChart\b.*?(?=^(?:function|const|let) |\Z)",
            app, re.M | re.S).group(0)
        cls.dates = [(date(2024, 1, 1) + timedelta(weeks=i)).isoformat() for i in range(31)]

    def browser(self, body, *, portfolio=None, ic=None, excluded_periods=None,
                performance=None, evaluation_window=None, annualization=52, chart=False):
        fixtures = {"dates": self.dates, "portfolio": portfolio or [], "ic": ic or [],
                    "excluded_periods": excluded_periods or [], "performance": performance or [],
                    "evaluation_window": evaluation_window or {}, "annualization": annualization}
        script = """
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const elements={},captures={};
const document={getElementById:id=>elements[id]||(elements[id]={
 textContent:'',innerHTML:'',querySelectorAll:()=>[],querySelector:()=>({
 getBoundingClientRect:()=>({left:0,width:760})})})};
const ANN=input.annualization,ROLL=ANN,PERIOD='周',CFG={metrics:{nw_lag:13}},META={variants:{}};
let START='';
const S={range:'3',ic:'spearman',variant:'raw',trend:'cumulative',annual:'Q10'};
let F={portfolio:input.portfolio,ic:input.ic,excluded_periods:input.excluded_periods,
 performance:input.performance,evaluation_window:input.evaluation_window};
const D={benchmark:input.portfolio},periodCache=new Map(),profileCache=new Map();
let DATES=input.dates,ALLDATES=[],VS=['raw'];
function lineChart(id,series){captures[id]=series}
function color(){return ''}
function label(v){return v}
function hideTip(){}
""" + self.source + (self.chart_source if chart else "") + "\nconsole.log(JSON.stringify((()=>{\n" + body + "\n})()));"
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

    def test_legacy_missing_signal_periods_do_not_hide_valid_nav_or_annual_return(self):
        rows = self.portfolio([None, .1, .2], [None, .02, .03])
        for i, row in enumerate(rows):
            row.update(n_stocks=0 if i == 0 else 10,
                       reason="missing_signal" if i == 0 else "")
        saved = [{"variant": "raw", "portfolio": "Q10", "annual_return": None,
                  "total_return": None, "status": "incomplete"}]
        actual = self.browser("""
S.range='all';return {rows:pRows(),nav:compound(pRows()),
 excess:compound(pRows(),'excess_return'),perf:perf('Q10'),benchmark:benchmark()};
""", portfolio=rows, performance=saved)
        self.assertEqual(len(actual["rows"]), 2)
        self.assertEqual(len(actual["benchmark"]), 2)
        self.assertEqual(actual["nav"][0], {"date": self.dates[1], "value": 1})
        self.assertAlmostEqual(actual["nav"][-1]["value"], 1.32)
        self.assertAlmostEqual(actual["excess"][-1]["value"], 1.02 * 1.03)
        self.assertAlmostEqual(actual["perf"]["total"], .32)
        self.assertAlmostEqual(actual["perf"]["annual"], 1.32 ** 26 - 1)

    def test_explicit_period_exclusions_apply_to_portfolio_ic_and_benchmark(self):
        rows = self.portfolio([.1, .2, .3])
        other = [dict(row, variant="industry") for row in rows]
        ic = self.ic_rows()[:3]
        actual = self.browser("""
return {raw:pRows().map(r=>r.entry),other:pRows('Q10','industry').map(r=>r.entry),
 ic:icRows().map(r=>r.entry),benchmark:benchmark().map(r=>r.entry)};
""", portfolio=rows + other, ic=ic,
            excluded_periods=[{"variant": "raw", "entry": self.dates[1]}])
        expected = [self.dates[0], self.dates[2]]
        self.assertEqual(actual["raw"], expected)
        self.assertEqual(actual["ic"], expected)
        self.assertEqual(set(actual["benchmark"]), set(expected))
        self.assertEqual(actual["other"], self.dates[:3])

    def test_missing_held_returns_are_preserved_instead_of_filtered(self):
        rows = self.portfolio([.1, None, .2])
        for row in rows:
            row.update(n_stocks=10, reason="missing_held_return" if row["return"] is None else "")
        actual = self.browser("return {n:pRows().length,perf:perf('Q10'),points:compound(pRows())};",
                              portfolio=rows)
        self.assertEqual(actual["n"], 3)
        self.assertIsNone(actual["perf"]["annual"])
        self.assertIsNone(actual["perf"]["total"])
        self.assertEqual([p["value"] for p in actual["points"]], [1, 1.1, None, 1, 1.2])

    def test_empty_cash_group_with_an_available_signal_is_retained(self):
        rows = self.portfolio([0., 0.])
        for row in rows:
            row.update(n_stocks=0, reason="empty_group_cash")
        all_rows = [dict(row, portfolio="ALL", n_stocks=5, reason="") for row in rows]
        actual = self.browser("return {n:pRows().length,total:perf('Q10').total};",
                              portfolio=rows + all_rows)
        self.assertEqual(actual, {"n": 2, "total": 0})

    def test_nav_restarts_after_absent_period_but_does_not_bridge_the_gap(self):
        rows = self.portfolio([.1, .5, .2])
        del rows[1]
        actual = self.browser("return {points:compound(pRows()),perf:perf('Q10')};", portfolio=rows)
        self.assertEqual([p["value"] for p in actual["points"]], [1, 1.1, None, 1, 1.2])
        self.assertIsNone(actual["perf"]["annual"])
        self.assertIsNone(actual["perf"]["dd"])

    def test_nav_does_not_restart_a_bankrupt_account(self):
        points = self.browser("return compound(pRows());", portfolio=self.portfolio([.1, -1, .5]))
        self.assertEqual([p["value"] for p in points], [1, 1.1, 0])

    def test_chart_axis_trims_missing_edges_and_preserves_internal_breaks(self):
        actual = self.browser("""
lineChart('chart',[{name:'IC',points:[{date:'2013-01-01',value:null},
 {date:'2020-01-01',value:.1},{date:'2021-01-01',value:null},
 {date:'2022-01-01',value:.2},{date:'2026-01-01',value:null}]}]);
return elements.chart.innerHTML;
""", chart=True)
        self.assertNotIn("2013.01", actual)
        self.assertNotIn("2026.01", actual)
        self.assertIn("2020.01", actual)
        self.assertIn("2022.01", actual)
        path = re.search(r'<path[^>]+ d="([^"]*)"', actual).group(1)
        self.assertEqual(path.count("M"), 2)
        self.assertNotIn("L", path)

    def test_monthly_rolling_ic_requires_twelve_observations_and_restarts_after_missing(self):
        rows = self.ic_rows()
        for i, row in enumerate(rows):
            row["exit"] = self.dates[i + 1]
            row["ic"] = None if i in (0, 13) else .1
        points = self.browser("return rolling(icRows());", ic=rows, annualization=12)
        self.assertTrue(all(point["value"] is None for point in points[:12]))
        self.assertAlmostEqual(points[12]["value"], .1)
        self.assertTrue(all(point["value"] is None for point in points[13:25]))
        self.assertAlmostEqual(points[25]["value"], .1)

    def test_rolling_ic_does_not_join_windows_across_an_absent_period(self):
        rows = self.ic_rows()
        for i, row in enumerate(rows):
            row.update(exit=self.dates[i + 1], ic=.1)
        del rows[12]
        points = self.browser("return rolling(icRows());", ic=rows, annualization=12)
        self.assertAlmostEqual(points[11]["value"], .1)
        self.assertTrue(all(point["value"] is None for point in points[12:23]))
        self.assertAlmostEqual(points[23]["value"], .1)

    def test_rolling_chart_starts_at_first_complete_window(self):
        actual = self.browser("""
lineChart('chart',[{name:'滚动IC',points:rolling(icRows())}]);return elements.chart.innerHTML;
""", ic=self.ic_rows(), annualization=12, chart=True)
        self.assertNotIn("2024.01", actual)
        self.assertNotIn("2024.02", actual)
        self.assertIn("2024.03", actual)

    def test_missing_signal_is_disclosed_with_actual_holding_window(self):
        rows = self.portfolio([None, .1, .2])
        window = {"requested_start": "2013-01-04", "requested_end": "2026-08-20",
                  "variants": {"raw": {"holding_start": self.dates[1],
                  "holding_end": self.dates[3], "included_periods": 2,
                  "excluded_periods": 1, "excluded_years": ["2013", "2014"]}}}
        actual = self.browser("return evaluationDisclosure();", portfolio=rows,
                              excluded_periods=[{"variant": "raw", "entry": self.dates[0],
                                                 "reason": "missing_signal"}],
                              evaluation_window=window)
        self.assertIn("2013-01-04", actual)
        self.assertIn(self.dates[1], actual)
        self.assertIn(self.dates[3], actual)
        self.assertIn("2013、2014", actual)
        self.assertIn("部分年份保留有效月份或周", actual)
        self.assertIn("无信号期间不作为现金收益", actual)

    def test_refresh_preserves_variants_with_no_signal_anywhere_in_the_window(self):
        actual = self.browser("""
F.name='factor';D.factors=[F];META.variants={raw:'raw',industry:'industry'};
S.range='all';refresh();const rawDates=DATES.slice();
S.variant='industry';refresh();return {variants:VS,rawDates,emptyDates:DATES};
""", portfolio=self.portfolio([.1, .2]),
            evaluation_window={"variants": {"industry": {"included_periods": 0}}})
        self.assertEqual(actual["variants"], ["raw", "industry"])
        self.assertEqual(actual["rawDates"], self.dates[:2])
        self.assertEqual(actual["emptyDates"], [])

    def test_newey_west_does_not_bridge_a_signal_period_removed_from_the_date_axis(self):
        rows = self.ic_rows()
        for i, row in enumerate(rows):
            row["exit"] = self.dates[i + 1]
        del rows[5]
        actual = self.browser("DATES=icRows().map(r=>r.entry);return icStats(icRows());", ic=rows)
        self.assertEqual(actual["n"], len(rows))
        self.assertIsNone(actual["nw"])
        self.assertIsNone(actual["low"])
