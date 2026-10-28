"""Explicit turnover diagnostics and arithmetic excess-Sharpe definitions."""
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.backtest import evaluate_factor, performance_metrics


def fixture():
    entries = pd.to_datetime(["2015-12-21", "2015-12-28", "2016-01-04", "2016-01-11"])
    periods = pd.DataFrame({"entry": entries[:-1], "exit": entries[1:]})
    signals, labels = [], []
    for t, row in enumerate(periods.itertuples(index=False)):
        for j in range(10):
            rank = j + 1 if t == 0 else 10 - j
            code = f"{j:06d}.SZ"
            signals.append({"entry": row.entry, "exit": row.exit, "code": code,
                            "variant": "raw", "signal": float(rank),
                            "percentile": rank / 10., "group": rank})
            labels.append({"entry": row.entry, "exit": row.exit, "code": code,
                           "return": .01 * (j + 1) + .005 * t * (j % 3),
                           "unresolved": False, "trade_restricted": False})
    return (pd.DataFrame(signals), pd.DataFrame(labels),
            periods.assign(**{"return": [.02, -.01, .03]}), periods)


class TurnoverSharpeTests(unittest.TestCase):
    def test_sharpe_uses_arithmetic_mean_and_requested_frequency(self):
        returns = np.array([.1, -.03, .02])
        excess = np.array([.06, -.02, .01])
        for frequency in (12, 52):
            with self.subTest(frequency=frequency):
                actual = performance_metrics(returns, excess_returns=excess, annualization=frequency)
                self.assertAlmostEqual(actual["sharpe_ratio"],
                                       np.sqrt(frequency) * returns.mean() / returns.std(ddof=1))
                self.assertAlmostEqual(actual["excess_sharpe"],
                                       np.sqrt(frequency) * excess.mean() / excess.std(ddof=1))
                self.assertEqual(actual["sharpe_n_periods"], 3)
                self.assertEqual(actual["excess_sharpe_n_periods"], 3)
                # Preserve the older geometric-return ratio separately.
                self.assertAlmostEqual(actual["information_ratio"],
                                       actual["annual_return"] / actual["annual_volatility"])
                self.assertNotAlmostEqual(actual["information_ratio"], actual["sharpe_ratio"])

    def test_constant_short_and_missing_observations_do_not_make_infinite_sharpe(self):
        for values in ([], [.1], [.1, .1, .1], [np.nan, .1, np.nan]):
            with self.subTest(values=values):
                actual = performance_metrics(values, excess_returns=values)
                self.assertTrue(np.isnan(actual["sharpe_ratio"]))
                self.assertTrue(np.isnan(actual["excess_sharpe"]))
        actual = performance_metrics([.1, np.nan, -.1])
        self.assertEqual(actual["sharpe_n_periods"], 2)
        self.assertEqual(actual["sharpe_ratio"], 0.)
        self.assertEqual(actual["status"], "incomplete")
        self.assertTrue(np.isnan(actual["annual_return"]))

    def test_excess_below_minus_one_is_not_actual_portfolio_bankruptcy(self):
        excess = np.array([-1.2, .1, .2])
        actual = performance_metrics([.1, .1, .2], excess_returns=excess)
        self.assertEqual(actual["excess_sharpe_n_periods"], 3)
        self.assertAlmostEqual(actual["excess_sharpe"],
                               np.sqrt(52) * excess.mean() / excess.std(ddof=1))
        stopped = performance_metrics([.1, -1., .2], excess_returns=excess)
        self.assertEqual(stopped["excess_sharpe_n_periods"], 2)

    def test_initial_build_excluded_from_warning_and_missing_turnover_not_zero(self):
        actual = performance_metrics([.01] * 4, turnover=[1., 1.2, np.nan, .4],
                                     initial_build=[True, False, False, False],
                                     turnover_warning_threshold=.5)
        self.assertEqual(actual["initial_build_periods"], 1)
        self.assertEqual(actual["turnover_observations"], 2)
        self.assertEqual(actual["turnover_missing_periods"], 1)
        self.assertAlmostEqual(actual["mean_turnover_one_way"], .4)
        self.assertAlmostEqual(actual["p95_turnover_one_way"], .58)
        self.assertEqual(actual["high_turnover_fraction"], .5)
        self.assertTrue(np.isnan(actual["annual_turnover"]))

    def test_invalid_threshold_and_diagnostic_grids_fail_explicitly(self):
        for invalid in (True, -1, np.nan, np.inf, "0.5"):
            with self.subTest(threshold=invalid), self.assertRaises(ValueError):
                performance_metrics([.1, .2], turnover_warning_threshold=invalid)
        with self.assertRaisesRegex(ValueError, "initial-build"):
            performance_metrics([.1, .2], initial_build=[True])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            performance_metrics([.1, .2], turnover=[1., -.1])

    def test_period_diagnostics_and_annual_initial_flag_are_not_reset_at_new_year(self):
        tables = evaluate_factor(*fixture(), turnover_warning_threshold=.5)
        rows = tables["portfolio_weekly"]
        high = rows[rows.portfolio.eq("Q10")].sort_values("entry")
        spread = rows[rows.portfolio.eq("Q10-Q1")].sort_values("entry")
        np.testing.assert_allclose(high.turnover, [1., 2., 0.])
        np.testing.assert_allclose(high.turnover_one_way, [.5, 1., 0.])
        np.testing.assert_allclose(high.target_turnover_one_way, [.5, 1., 0.])
        self.assertEqual(high.initial_build.tolist(), [True, False, False])
        self.assertTrue(pd.isna(high.high_turnover.iloc[0]))
        self.assertEqual(high.high_turnover.iloc[1:].tolist(), [True, False])
        np.testing.assert_allclose(spread.turnover_one_way, [1., 2., 0.])
        annual = tables["annual_performance"].set_index(["portfolio", "year"])
        self.assertEqual(annual.loc[("Q10", 2015), "initial_build_periods"], 1)
        self.assertEqual(annual.loc[("Q10", 2016), "initial_build_periods"], 0)
        self.assertEqual(annual.loc[("Q10", 2016), "turnover_observations"], 1)

    def test_long_short_excess_sharpe_cancels_benchmark_without_double_subtraction(self):
        inputs = fixture()
        tables = evaluate_factor(*inputs)
        performance = tables["performance"].set_index("portfolio")
        rows = tables["portfolio_weekly"]
        high = rows[rows.portfolio.eq("Q10")].sort_values("entry")
        low = rows[rows.portfolio.eq("Q1")].sort_values("entry")
        excess = high.excess_return.to_numpy() - low.excess_return.to_numpy()
        expected = np.sqrt(52) * excess.mean() / excess.std(ddof=1)
        self.assertAlmostEqual(performance.loc["Q10-Q1", "excess_sharpe"], expected)
        self.assertAlmostEqual(performance.loc["Q10-Q1", "sharpe_ratio"], expected)
        self.assertAlmostEqual(performance.loc["Q10", "excess_sharpe"],
                               np.sqrt(52) * high.excess_return.mean() / high.excess_return.std(ddof=1))
        self.assertTrue(np.isnan(performance.loc["Q10-Q1", "excess_total_return"]))
        changed_benchmark = inputs[2].copy()
        changed_benchmark["return"] = [.4, -.2, .3]
        changed = evaluate_factor(inputs[0], inputs[1], changed_benchmark, inputs[3])["performance"].set_index("portfolio")
        self.assertEqual(changed.loc["Q10-Q1", "excess_sharpe"], expected)
        self.assertNotAlmostEqual(changed.loc["Q10", "excess_sharpe"], performance.loc["Q10", "excess_sharpe"])

    def test_missing_held_return_leaves_next_turnover_warning_unknown(self):
        signals, labels, benchmark, periods = fixture()
        bad = labels.entry.eq("2015-12-28") & labels.code.eq("000000.SZ")
        labels.loc[bad, "return"] = np.nan
        labels.loc[bad, "unresolved"] = True
        tables = evaluate_factor(signals, labels, benchmark, periods, missing_returns="available_case")
        high = tables["portfolio_weekly"].query("portfolio == 'Q10'").sort_values("entry")
        self.assertTrue(np.isnan(high.turnover.iloc[2]))
        self.assertTrue(pd.isna(high.high_turnover.iloc[2]))
        summary = tables["performance"].set_index("portfolio").loc["Q10"]
        self.assertEqual(summary.turnover_observations, 1)
        self.assertEqual(summary.turnover_missing_periods, 1)
        self.assertEqual(summary.high_turnover_fraction, 1.)


if __name__ == "__main__":
    unittest.main()
