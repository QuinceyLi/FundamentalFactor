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

    def test_missing_entire_signal_week_does_not_infer_cash_or_next_target_change(self):
        signals, labels, benchmark, weeks = fixture()
        middle, last = weeks.entry.iloc[1:]
        signals = signals[~signals.entry.eq(middle)].copy()
        result = evaluate(signals, labels, benchmark, weeks)
        q1 = result["portfolio_weekly"].query("portfolio == 'Q1'").sort_values("entry")
        self.assertTrue(np.isnan(q1["return"].iloc[1]))
        self.assertEqual(q1.reason.iloc[1], "missing_signal")
        self.assertTrue(np.isnan(q1.cash_weight.iloc[1]))
        self.assertTrue(q1.target_turnover.iloc[1:].isna().all())
        self.assertTrue(q1.turnover.iloc[1:].isna().all())
        self.assertTrue(result["holdings"].query("entry == @middle").empty)
        self.assertTrue(result["holdings"].query("entry == @last and portfolio == 'Q1'").drifted_weight.isna().all())
        perf = result["performance"].query("portfolio == 'Q1'").iloc[0]
        self.assertTrue(np.isnan(perf.annual_target_turnover))

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
