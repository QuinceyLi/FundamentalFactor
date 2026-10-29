"""Available-case factor statistics retain frozen signals and disclose estimation."""
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.backtest import evaluate_factor


def fixture():
    weeks = pd.DataFrame({"entry": pd.date_range("2020-01-06", periods=3, freq="7D")})
    weeks["exit"] = weeks.entry + pd.Timedelta(days=7)
    signals, labels = [], []
    for t, week in weeks.iterrows():
        for i in range(40):
            base = {"entry": week.entry, "exit": week.exit, "code": f"{i:06d}.SZ"}
            signals.append(base | {"variant": "raw", "signal": float(i),
                                   "percentile": (i + 1) / 40, "group": i // 4 + 1})
            labels.append(base | {"return": (i + t + 1) * .001, "unresolved": False,
                                  "trade_restricted": False, "trade_restriction_unknown": False})
    benchmark = weeks.assign(**{"return": [.0205, .0215, .0225]})
    return pd.DataFrame(signals), pd.DataFrame(labels), benchmark, weeks


def evaluate(signals, labels, benchmark, weeks, **kwargs):
    return evaluate_factor(signals, labels, benchmark, weeks,
                           missing_returns="available_case", empty_groups="cash", **kwargs)


class AvailableReturnTests(unittest.TestCase):
    def test_partial_labels_change_evaluation_weights_not_signal_weights(self):
        signals, labels, benchmark, weeks = fixture()
        first = weeks.entry.iloc[0]
        missing = labels.entry.eq(first) & labels.code.eq("000000.SZ")
        labels.loc[missing, ["return", "unresolved"]] = [np.nan, True]
        benchmark.loc[0, "return"] = labels.loc[labels.entry.eq(first), "return"].mean()
        benchmark["unresolved"] = [True, False, False]
        frozen_signals, original_labels = signals.copy(deep=True), labels.copy(deep=True)
        result = evaluate(signals, labels, benchmark, weeks)
        pd.testing.assert_frame_equal(signals, frozen_signals)
        pd.testing.assert_frame_equal(labels, original_labels)
        row = result["portfolio_weekly"].query("portfolio == 'Q1' and entry == @first").iloc[0]
        self.assertAlmostEqual(row["return"], .003)
        self.assertAlmostEqual(row.missing_weight, .25)
        self.assertTrue(row.complete)
        self.assertFalse(row.data_complete)
        self.assertEqual(row.reason, "available_case_partial")
        self.assertEqual(row.return_method, "available_case")
        self.assertEqual(row.evaluation_weight_sum, 1.)
        self.assertAlmostEqual(row.excess_return, .003 - benchmark["return"].iloc[0])
        ledger = result["holdings"].query("portfolio == 'Q1' and entry == @first")
        self.assertEqual(len(ledger), 4)
        self.assertTrue(ledger.target_weight.eq(.25).all())
        self.assertEqual(ledger.loc[ledger.code.eq("000000.SZ"), "evaluation_weight"].iloc[0], 0.)
        self.assertTrue(ledger.loc[ledger.code.eq("000000.SZ"), "return"].isna().all())
        np.testing.assert_allclose(ledger.loc[ledger["return"].notna(), "evaluation_weight"], 1 / 3)
        self.assertAlmostEqual((ledger.evaluation_weight * ledger["return"]).sum(), row["return"])
        ic = result["ic_weekly"].query("entry == @first and subset == 'ALL' and method == 'spearman'").iloc[0]
        self.assertEqual((ic.n_stocks, ic.n_labels), (40, 39))
        q1 = result["portfolio_weekly"].query("portfolio == 'Q1'").sort_values("entry")
        self.assertTrue(q1.nav.notna().all())
        self.assertEqual(q1.segment_id.nunique(), 1)
        self.assertTrue(np.isnan(q1.turnover.iloc[1]))
        np.testing.assert_allclose(q1.target_turnover, [1., 0., 0.])
        perf = result["performance"].query("portfolio == 'Q1'").iloc[0]
        self.assertTrue(np.isfinite(perf.annual_return))
        self.assertTrue(np.isnan(perf.annual_turnover))
        self.assertAlmostEqual(perf.annual_target_turnover, 52 / 3)

    def test_all_labels_missing_is_not_an_empty_cash_group(self):
        signals, labels, benchmark, weeks = fixture()
        first = weeks.entry.iloc[0]
        labels.loc[labels.entry.eq(first) & labels.code.lt("000004.SZ"), ["return", "unresolved"]] = [np.nan, True]
        result = evaluate(signals, labels, benchmark, weeks)
        row = result["portfolio_weekly"].query("portfolio == 'Q1' and entry == @first").iloc[0]
        self.assertEqual(row.n_stocks, 4)
        self.assertEqual(row.n_labels, 0)
        self.assertFalse(row.complete)
        self.assertEqual(row.cash_weight, 0.)
        self.assertEqual(row.evaluation_weight_sum, 0.)
        self.assertTrue(np.isnan(row["return"]))
        self.assertEqual(row.reason, "missing_held_return")

    def test_empty_quantile_sells_to_cash_then_rebuilds(self):
        signals, labels, benchmark, weeks = fixture()
        middle = weeks.entry.iloc[1]
        signals.loc[signals.entry.eq(middle) & signals.group.eq(1), "group"] = 2
        result = evaluate(signals, labels, benchmark, weeks)
        q1 = result["portfolio_weekly"].query("portfolio == 'Q1'").sort_values("entry")
        self.assertEqual(q1["return"].iloc[1], 0.)
        self.assertEqual(q1.cash_weight.iloc[1], 1.)
        self.assertEqual(q1.return_method.iloc[1], "cash")
        self.assertTrue(q1.complete.all())
        self.assertTrue(q1.data_complete.all())
        np.testing.assert_allclose(q1.turnover, [1., 1., 1.])
        np.testing.assert_allclose(q1.target_turnover, [1., 1., 1.])
        self.assertAlmostEqual(q1.nav.iloc[1], q1.nav.iloc[0])
        self.assertAlmostEqual(q1.nav.iloc[-1], (1 + q1["return"].iloc[0]) * (1 + q1["return"].iloc[-1]))
        exits = result["holdings"].query("entry == @middle and portfolio == 'Q1'")
        self.assertEqual(len(exits), 4)
        self.assertTrue(exits.is_exit.all())
        self.assertTrue(exits.target_weight.eq(0).all())
        self.assertAlmostEqual(exits.trade_weight.sum(), -1.)
        high = result["portfolio_weekly"].query("entry == @middle and portfolio == 'Q10'").iloc[0]
        spread = result["portfolio_weekly"].query("entry == @middle and portfolio == 'Q10-Q1'").iloc[0]
        self.assertAlmostEqual(spread["return"], high["return"])
        self.assertEqual(spread.cash_weight, .5)
        self.assertEqual(spread.trade_restricted_fraction, 0.)

    def test_both_empty_legs_have_cash_return_and_finite_diagnostics(self):
        signals, labels, benchmark, weeks = fixture()
        first = weeks.entry.iloc[0]
        signals.loc[signals.entry.eq(first), "group"] = 5
        result = evaluate(signals, labels, benchmark, weeks)
        spread = result["portfolio_weekly"].query("entry == @first and portfolio == 'Q10-Q1'").iloc[0]
        self.assertTrue(spread.complete)
        self.assertEqual(spread["return"], 0.)
        self.assertEqual(spread.cash_weight, 1.)
        self.assertEqual(spread.evaluation_weight_sum, 0.)
        self.assertEqual(spread.trade_restricted_fraction, 0.)
        self.assertEqual(spread.trade_restriction_unknown_fraction, 0.)

    def test_multiple_cash_weeks_preserve_capital_for_reentry(self):
        signals, labels, benchmark, weeks = fixture()
        signals.loc[signals.entry.lt(weeks.entry.iloc[-1]) & signals.group.eq(1), "group"] = 2
        result = evaluate(signals, labels, benchmark, weeks)
        q1 = result["portfolio_weekly"].query("portfolio == 'Q1'").sort_values("entry")
        self.assertTrue(q1.complete.all())
        np.testing.assert_allclose(q1["return"].iloc[:2], 0.)
        np.testing.assert_allclose(q1.nav.iloc[:2], 1.)
        np.testing.assert_allclose(q1.turnover, [0., 0., 1.])
        np.testing.assert_allclose(q1.target_turnover, [0., 0., 1.])
        self.assertGreater(q1.nav.iloc[-1], 1.)

    def test_missing_entire_signal_week_is_excluded_and_restarts_a_separate_segment(self):
        signals, labels, benchmark, weeks = fixture()
        middle, last = weeks.entry.iloc[1:]
        signals = signals[~signals.entry.eq(middle)].copy()
        result = evaluate(signals, labels, benchmark, weeks)
        q1 = result["portfolio_weekly"].query("portfolio == 'Q1'").sort_values("entry")
        self.assertEqual(len(q1), 2)
        self.assertFalse(q1.entry.eq(middle).any())
        self.assertTrue(q1["return"].notna().all())
        self.assertEqual(q1.segment_id.tolist(), [1, 2])
        self.assertEqual(q1.initial_build.tolist(), [True, True])
        np.testing.assert_allclose(q1.target_turnover, [1., 1.])
        np.testing.assert_allclose(q1.turnover, [1., 1.])
        np.testing.assert_allclose(q1.nav, 1 + q1["return"])
        self.assertTrue(result["holdings"].query("entry == @middle").empty)
        self.assertTrue(result["holdings"].query("entry == @last and portfolio == 'Q1'").drifted_weight.eq(0).all())
        excluded = result["excluded_periods"]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded.iloc[0].entry, middle)
        self.assertEqual(excluded.iloc[0].reason, "missing_signal")
        self.assertEqual(excluded.iloc[0].n_stocks, 0)
        perf = result["performance"].query("portfolio == 'Q1'").iloc[0]
        self.assertEqual(perf.status, "incomplete")
        self.assertTrue(np.isnan(perf.annual_return))
        self.assertTrue(np.isnan(perf.max_drawdown))
        self.assertTrue(np.isnan(perf.annual_target_turnover))

    def test_leading_and_trailing_missing_signal_periods_do_not_poison_active_statistics(self):
        signals, labels, benchmark, weeks = fixture()
        first, middle, last = weeks.entry
        for missing in (first, last):
            with self.subTest(missing=missing):
                selected = signals[~signals.entry.eq(missing)]
                result = evaluate(selected, labels, benchmark, weeks, nw_lag=0)
                for portfolio in ("Q1", "Q10", "Q10-Q1"):
                    rows = result["portfolio_weekly"].query("portfolio == @portfolio").sort_values("entry")
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(rows.nav.notna().all())
                    self.assertEqual(rows.segment_id.nunique(), 1)
                    self.assertEqual(rows.initial_build.tolist(), [True, False])
                    self.assertEqual(rows.turnover.iloc[0], 2. if portfolio == "Q10-Q1" else 1.)
                    self.assertEqual(rows.target_turnover.iloc[0], 2. if portfolio == "Q10-Q1" else 1.)
                    perf = result["performance"].query("portfolio == @portfolio").iloc[0]
                    self.assertEqual(perf.status, "ok")
                    self.assertTrue(np.isfinite(perf.annual_return))
                self.assertFalse(result["ic_weekly"].entry.eq(missing).any())
                ic = result["ic_summary"].query("subset == 'ALL' and method == 'spearman'").iloc[0]
                self.assertEqual(ic.n, 2)
                self.assertEqual(ic.nw_status, "ok")

    def test_independent_variants_use_their_own_available_periods_and_keep_empty_variant_statistics(self):
        signals, labels, benchmark, weeks = fixture()
        delayed = signals[signals.entry.eq(weeks.entry.iloc[-1])].assign(variant="industry")
        signals = pd.concat([signals, delayed], ignore_index=True)
        result = evaluate(signals, labels, benchmark, weeks, variants=["raw", "industry", "size"],
                          sample_policy="independent")
        rows = result["portfolio_weekly"].query("portfolio == 'Q10'")
        self.assertEqual(rows.groupby("variant").size().to_dict(), {"industry": 1, "raw": 3})
        delayed_row = rows[rows.variant.eq("industry")].iloc[0]
        self.assertTrue(delayed_row.initial_build)
        self.assertEqual(delayed_row.turnover, 1.)
        self.assertEqual(delayed_row.target_turnover, 1.)
        self.assertEqual(result["excluded_periods"].groupby("variant").size().to_dict(), {"industry": 2, "size": 3})
        absent = result["performance"].query("variant == 'size'")
        self.assertEqual(len(absent), 12)
        self.assertTrue(absent.n_periods.eq(0).all())
        self.assertTrue(absent.status.eq("no_data").all())
        self.assertTrue(result["ic_summary"].query("variant == 'size'").n.eq(0).all())

    def test_inference_does_not_compress_missing_calendar_periods(self):
        signals, labels, benchmark, weeks = fixture()
        middle = weeks.entry.iloc[1]
        result = evaluate(signals[~signals.entry.eq(middle)], labels, benchmark, weeks, nw_lag=0)
        ic = result["ic_summary"].query("subset == 'ALL' and method == 'spearman'").iloc[0]
        self.assertEqual(ic.n, 2)
        self.assertEqual(ic.nw_status, "missing_calendar_weeks")
        self.assertTrue(pd.isna(ic.nw_lower))

    def test_unknown_held_returns_remain_in_active_periods(self):
        signals, labels, benchmark, weeks = fixture()
        middle = weeks.entry.iloc[1]
        labels.loc[labels.entry.eq(middle), ["return", "unresolved"]] = [np.nan, True]
        result = evaluate(signals, labels, benchmark, weeks)
        rows = result["portfolio_weekly"].query("portfolio == 'Q10'")
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows.loc[rows.entry.eq(middle), "return"].isna().all())
        self.assertTrue(result["excluded_periods"].empty)

    def test_bankruptcy_remains_terminal_after_an_excluded_signal_period(self):
        signals, labels, benchmark, weeks = fixture()
        first, middle, last = weeks.entry
        labels.loc[labels.entry.eq(first) & labels.code.lt("000004.SZ"), "return"] = -1.
        result = evaluate(signals[~signals.entry.eq(middle)], labels, benchmark, weeks)
        rows = result["portfolio_weekly"].query("portfolio == 'Q1'").sort_values("entry")
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows.nav.eq(0).all())
        self.assertTrue(np.isnan(rows["return"].iloc[-1]))
        self.assertFalse(rows.initial_build.iloc[-1])
        self.assertEqual(rows.reason.iloc[-1], "portfolio_bankruptcy")

    def test_empty_years_are_excluded_but_each_observed_partial_year_keeps_annual_statistics(self):
        source_signals, source_labels, _, _ = fixture()
        boundaries = pd.date_range("2019-12-01", "2021-03-01", freq="MS")
        weeks = pd.DataFrame({"entry": boundaries[:-1], "exit": boundaries[1:]})
        signals, labels = [], []
        for period in weeks.itertuples(index=False):
            signals.append(source_signals[source_signals.entry.eq(source_signals.entry.min())].assign(
                entry=period.entry, exit=period.exit))
            labels.append(source_labels[source_labels.entry.eq(source_labels.entry.min())].assign(
                entry=period.entry, exit=period.exit))
        signals = pd.concat(signals, ignore_index=True)
        signals = signals[~signals.entry.dt.year.eq(2020)]
        labels = pd.concat(labels, ignore_index=True)
        result = evaluate(signals, labels, weeks.assign(**{"return": .01}), weeks)
        rows = result["portfolio_weekly"].query("portfolio == 'Q10'").sort_values("entry")
        self.assertEqual(rows.entry.dt.year.tolist(), [2019, 2021, 2021])
        self.assertEqual(rows.segment_id.tolist(), [1, 2, 2])
        self.assertEqual(rows.initial_build.tolist(), [True, True, False])
        self.assertEqual(len(result["excluded_periods"]), 12)
        annual = result["annual_performance"].query("portfolio == 'Q10'")
        self.assertEqual(annual.year.tolist(), [2019, 2021])
        self.assertTrue(annual.annual_return.notna().all())
        self.assertTrue(annual.status.eq("ok").all())
        self.assertTrue(np.isnan(result["performance"].query("portfolio == 'Q10'").iloc[0].annual_return))

    def test_preexisting_calendar_gap_also_resets_the_position_segment(self):
        signals, labels, benchmark, weeks = fixture()
        middle = weeks.entry.iloc[1]
        signals, labels, benchmark, weeks = [frame[~frame.entry.eq(middle)]
                                            for frame in (signals, labels, benchmark, weeks)]
        result = evaluate(signals, labels, benchmark, weeks)
        rows = result["portfolio_weekly"].query("portfolio == 'Q10'").sort_values("entry")
        self.assertEqual(rows.segment_id.tolist(), [1, 2])
        self.assertEqual(rows.initial_build.tolist(), [True, True])
        self.assertEqual(rows.turnover.tolist(), [1., 1.])
        self.assertTrue(np.isnan(result["performance"].query("portfolio == 'Q10'").iloc[0].annual_return))

    def test_strict_benchmark_does_not_require_labels_for_excluded_signal_periods(self):
        signals, labels, benchmark, weeks = fixture()
        first = weeks.entry.iloc[0]
        signals = signals[~signals.entry.eq(first)]
        benchmark.loc[benchmark.entry.eq(first), "return"] = np.nan
        result = evaluate_factor(signals, labels, benchmark, weeks)
        rows = result["portfolio_weekly"].query("portfolio == 'Q10'")
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows.excess_nav.notna().all())

    def test_legacy_report_policy_keeps_missing_return_semantics(self):
        signals, labels, benchmark, weeks = fixture()
        first = weeks.entry.iloc[0]
        labels.loc[labels.entry.eq(first) & labels.code.eq("000000.SZ"), "return"] = np.nan
        result = evaluate_factor(signals, labels, benchmark, weeks, missing_returns="report")
        q1 = result["portfolio_weekly"].query("entry == @first and portfolio == 'Q1'").iloc[0]
        self.assertTrue(np.isnan(q1["return"]))
        self.assertFalse(q1.complete)

    def test_terminal_total_loss_does_not_create_new_capital(self):
        signals, labels, benchmark, weeks = fixture()
        first = weeks.entry.iloc[0]
        labels.loc[labels.entry.eq(first) & labels.code.lt("000004.SZ"), "return"] = -1.
        result = evaluate(signals, labels, benchmark, weeks)
        q1 = result["portfolio_weekly"].query("portfolio == 'Q1'").sort_values("entry")
        self.assertEqual(q1["return"].iloc[0], -1.)
        self.assertTrue(q1["return"].iloc[1:].isna().all())
        self.assertTrue(q1.nav.eq(0).all())


if __name__ == "__main__":
    unittest.main()
