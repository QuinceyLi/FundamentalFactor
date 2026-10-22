"""Capital-loss edge cases for the weekly, frozen-position backtester."""
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.backtest import evaluate_factor, performance_metrics


def _fixture(first_low=0.0, first_high=-1.0):
    weeks = pd.DataFrame({"entry": pd.date_range("2020-01-06", periods=3, freq="7D")})
    weeks["exit"] = weeks.entry + pd.Timedelta(days=7)
    signals, labels = [], []
    for t, week in weeks.iterrows():
        for i in range(10):
            code = f"{i:06d}.SZ"
            signal = {"entry": week.entry, "exit": week.exit, "code": code,
                      "variant": "raw", "signal": float(i), "percentile": (i + 1) / 10,
                      "group": i + 1}
            signals.append(signal)
            value = (first_low if i == 0 else first_high if i == 9 else 0.0) if t == 0 else .5
            labels.append({"entry": week.entry, "exit": week.exit, "code": code,
                           "return": value, "unresolved": False, "trade_restricted": False})
    return pd.DataFrame(signals), pd.DataFrame(labels), weeks.assign(**{"return": 0.0}), weeks


class TerminalCapitalTests(unittest.TestCase):
    def test_exact_terminal_loss_has_defined_total_and_drawdown(self):
        result = performance_metrics([.1, -.2, -1.0], [1.0, .1, .1])
        self.assertEqual(result["status"], "bankruptcy")
        self.assertEqual(result["total_return"], -1.0)
        self.assertEqual(result["annual_return"], -1.0)
        self.assertEqual(result["max_drawdown"], 1.0)
        self.assertEqual(result["n_periods"], 3)
        self.assertTrue(np.isnan(result["annual_turnover"]))

    def test_later_returns_cannot_recover_zero_capital(self):
        result = performance_metrics([.1, -1.0, 100.0])
        self.assertEqual(result["n_periods"], 2)
        self.assertEqual(result["total_return"], -1.0)
        self.assertAlmostEqual(result["annual_volatility"], np.std([.1, -1], ddof=1) * np.sqrt(52))

    def test_missing_history_before_total_loss_stays_incomplete(self):
        result = performance_metrics([.1, np.nan, -1.0, 100.0])
        self.assertEqual(result["status"], "bankruptcy")
        self.assertTrue(np.isnan(result["total_return"]))
        self.assertTrue(np.isnan(result["annual_return"]))
        self.assertTrue(np.isnan(result["max_drawdown"]))

    def test_negative_equity_is_distinct_from_zero_capital(self):
        result = performance_metrics([.1, -1.2, 100.0])
        self.assertEqual(result["status"], "insolvency")
        self.assertEqual(result["n_periods"], 2)
        self.assertTrue(np.isnan(result["total_return"]))
        self.assertTrue(np.isnan(result["annual_return"]))
        self.assertTrue(np.isnan(result["max_drawdown"]))

    def test_runner_does_not_reinvest_after_whole_group_goes_to_zero(self):
        result = evaluate_factor(*_fixture())
        weekly = result["portfolio_weekly"].query("portfolio == 'Q10'").sort_values("entry")
        self.assertEqual(weekly["return"].iloc[0], -1)
        self.assertTrue(weekly["return"].iloc[1:].isna().all())
        self.assertTrue(weekly.turnover.iloc[1:].isna().all())
        self.assertTrue((weekly.nav == 0).all())
        self.assertTrue((weekly.reason.iloc[1:] == "portfolio_bankruptcy").all())
        ledger = result["holdings"].query("portfolio == 'Q10'")
        self.assertEqual(len(ledger), 1)
        summary = result["performance"].query("portfolio == 'Q10'").iloc[0]
        self.assertEqual(summary.total_return, -1)
        self.assertEqual(summary.annual_return, -1)
        self.assertEqual(summary.max_drawdown, 1)
        self.assertEqual(summary.n_periods, 1)

    def test_spread_insolvency_stops_spread_without_stopping_its_legs(self):
        result = evaluate_factor(*_fixture(first_low=.5, first_high=-.9))
        spread = result["portfolio_weekly"].query("portfolio == 'Q10-Q1'").sort_values("entry")
        self.assertAlmostEqual(spread["return"].iloc[0], -1.4)
        self.assertTrue(spread["return"].iloc[1:].isna().all())
        self.assertTrue(spread.nav.isna().all())
        high = result["portfolio_weekly"].query("portfolio == 'Q10'")
        self.assertTrue(high["return"].notna().all())
        summary = result["performance"].query("portfolio == 'Q10-Q1'").iloc[0]
        self.assertEqual(summary.status, "insolvency")
        self.assertTrue(np.isnan(summary.total_return))


if __name__ == "__main__":
    unittest.main()
