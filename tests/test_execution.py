import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import DataGateError
from fundamental_factor.execution import _Account, _mark, run_execution


def fixture(targets, codes=("A", "B", "C")):
    endpoints = pd.date_range("2026-01-05", periods=len(targets) + 1, freq="W-MON")
    weeks = pd.DataFrame({"entry": endpoints[:-1], "exit": endpoints[1:]})
    calendar = pd.bdate_range(endpoints[0], endpoints[-1])
    signals = pd.DataFrame([dict(entry=week.entry, exit=week.exit, code=code, variant="raw", group=10)
                            for week, group in zip(weeks.itertuples(), targets) for code in group],
                           columns=["entry", "exit", "code", "variant", "group"])
    quotes = pd.DataFrame([dict(date=date, code=code, open=10., close=10.) for date in calendar for code in codes])
    status = quotes[["date", "code"]].assign(suspended=False, limit_up=False, limit_down=False)
    actions = pd.DataFrame(columns=["date", "code", "share_ratio", "cash_per_share", "notice_date"])
    return dict(signals=signals, quotes=quotes, status=status, actions=actions, weeks=weeks,
                calendar=calendar, initial_capital=100., portfolios=["ALL"])


class ExecutionTests(unittest.TestCase):
    def test_suspended_exit_remains_held_until_next_weekly_rebalance(self):
        inputs = fixture([["A"], ["B"], ["B"]])
        week = inputs["weeks"].entry.iloc[1]
        status = inputs["status"]
        status.loc[status.code.eq("A") & status.date.between(week, week + pd.Timedelta(days=4)), "suspended"] = True
        result = run_execution(**inputs)
        second = result["holdings"].query("entry == @week").set_index("code")
        self.assertEqual(second.loc["A", "target_weight"], 0)
        self.assertEqual(second.loc["A", "actual_weight"], 1)
        self.assertEqual(second.loc["B", "actual_weight"], 0)
        self.assertEqual(result["weekly"].turnover.tolist(), [1., 0., 2.])
        later_trades = result["trades"].query("entry > @week")
        self.assertEqual(later_trades.query("code == 'A'").executed_shares.iloc[0], -10.)
        self.assertTrue((result["daily"].cash >= 0).all())

    def test_directional_limits_allow_opposite_side(self):
        for flag, blocked in (("limit_up", False), ("limit_down", True)):
            with self.subTest(sell_flag=flag):
                inputs = fixture([["A"], ["B"]])
                week = inputs["weeks"].entry.iloc[1]
                s = inputs["status"]
                s.loc[s.code.eq("A") & s.date.eq(week), flag] = True
                result = run_execution(**inputs)
                trade = result["trades"].query("entry == @week and code == 'A'").iloc[0]
                self.assertEqual(trade.executed_shares, 0 if blocked else -10.)
        for flag, blocked in (("limit_up", True), ("limit_down", False)):
            with self.subTest(buy_flag=flag):
                inputs = fixture([["A"]])
                inputs["status"].loc[inputs["status"].code.eq("A"), flag] = True
                result = run_execution(**inputs)
                self.assertEqual(result["trades"].executed_shares.iloc[0], 0 if blocked else 10.)

    def test_missing_quote_and_unknown_status_freeze_without_future_fill(self):
        inputs = fixture([["A"], ["B"]])
        week = inputs["weeks"].entry.iloc[1]
        q = inputs["quotes"]
        inputs["quotes"] = q[~(q.date.eq(week) & q.code.eq("A"))].copy()
        q = inputs["quotes"]
        q.loc[q.date.gt(week) & q.code.eq("A"), ["open", "close"]] = 20.
        result = run_execution(**inputs)
        held = result["holdings"].query("entry == @week and code == 'A'").iloc[0]
        self.assertEqual(held.mark, 10.)
        self.assertLess(held.mark_date, week)
        self.assertEqual(held.mark_source, "carry_missing_open")
        self.assertEqual(result["trades"].query("entry == @week and code == 'A'").rejection_reason.iloc[0], "missing_open")
        inputs = fixture([["A"]])
        inputs["status"]["suspended"] = pd.Series(pd.NA, index=inputs["status"].index, dtype="boolean")
        result = run_execution(**inputs)
        self.assertEqual(result["weekly"].cash_weight.iloc[0], 1.)
        self.assertEqual(result["trades"].rejection_reason.iloc[0], "unknown_status")

    def test_sells_fund_only_available_cash_and_buys_scale_proportionally(self):
        inputs = fixture([["A", "B"], ["C"]])
        week = inputs["weeks"].entry.iloc[1]
        s = inputs["status"]
        s.loc[s.code.eq("A") & s.date.eq(week), "limit_down"] = True
        result = run_execution(**inputs)
        held = result["holdings"].query("entry == @week").set_index("code")
        self.assertEqual(held.loc["A", "actual_weight"], .5)
        self.assertEqual(held.loc["C", "actual_weight"], .5)
        buy = result["trades"].query("entry == @week and code == 'C'").iloc[0]
        self.assertEqual(buy.executed_value, 50.)
        self.assertTrue(buy.cash_scaled)
        self.assertEqual(buy.rejection_reason, "cash_shortfall")
        self.assertTrue(np.allclose(result["daily"].nav, 100.))

    def test_dividend_and_split_during_suspension_preserve_value_not_turnover(self):
        inputs = fixture([["A"]])
        event = pd.Timestamp("2026-01-07")
        inputs["actions"] = pd.DataFrame([dict(date=event, code="A", share_ratio=1., cash_per_share=1., notice_date=event)])
        q = inputs["quotes"]
        q.loc[q.code.eq("A") & q.date.ge(event), ["open", "close"]] = 4.5
        q.loc[q.code.eq("A") & q.date.eq(event), ["open", "close"]] = np.nan
        s = inputs["status"]
        s.loc[s.code.eq("A") & s.date.eq(event), "suspended"] = True
        result = run_execution(**inputs)
        self.assertTrue(np.allclose(result["daily"].nav, 100.))
        self.assertEqual(result["daily"].dividend_cash.sum(), 10.)
        self.assertEqual(result["daily"].bonus_shares.sum(), 10.)
        self.assertEqual(result["weekly"]["return"].iloc[0], 0.)
        self.assertEqual(result["weekly"].turnover.iloc[0], 1.)
        self.assertEqual(len(result["trades"]), 1)

    def test_new_ex_date_purchase_does_not_receive_old_holder_distribution(self):
        inputs = fixture([["A"]])
        event = inputs["weeks"].entry.iloc[0]
        inputs["actions"] = pd.DataFrame([dict(date=event, code="A", share_ratio=1., cash_per_share=1., notice_date=event)])
        inputs["quotes"].loc[inputs["quotes"].code.eq("A"), ["open", "close"]] = 4.5
        result = run_execution(**inputs)
        self.assertEqual(result["daily"].dividend_cash.sum(), 0.)
        self.assertEqual(result["daily"].bonus_shares.sum(), 0.)
        self.assertAlmostEqual(result["weekly"]["return"].iloc[0], 0.)

    def test_terminal_zero_is_real_loss_and_not_suspension_or_a_trade(self):
        inputs = fixture([["A"]])
        event = pd.Timestamp("2026-01-07")
        inputs["terminal"] = pd.DataFrame([dict(date=event, code="A", cash_per_share=0., available_date=event)])
        inputs["quotes"] = inputs["quotes"][~(inputs["quotes"].code.eq("A") & inputs["quotes"].date.ge(event))]
        result = run_execution(**inputs)
        self.assertEqual(result["weekly"]["return"].iloc[0], -1.)
        self.assertEqual(result["performance"].status.iloc[0], "bankruptcy")
        self.assertEqual(result["daily"].iloc[-1].n_stocks, 0)
        self.assertEqual(len(result["trades"]), 1)

    def test_terminal_payment_enters_cash_and_future_settlement_information_rejected(self):
        inputs = fixture([["A"]])
        event = pd.Timestamp("2026-01-07")
        inputs["terminal"] = pd.DataFrame([dict(date=event, code="A", cash_per_share=2., available_date=event)])
        result = run_execution(**inputs)
        self.assertAlmostEqual(result["weekly"]["return"].iloc[0], -.8)
        self.assertEqual(result["daily"].terminal_cash.sum(), 20.)
        inputs["terminal"]["available_date"] = event + pd.Timedelta(days=1)
        with self.assertRaises(DataGateError):
            run_execution(**inputs)

    def test_unvalued_existing_inventory_is_error(self):
        account = _Account("raw", "ALL", 0., shares={"A": 10.})
        with self.assertRaises(DataGateError):
            _mark(account, pd.Timestamp("2026-01-05"), {}, {}, "open")

    def test_empty_target_goes_to_cash_and_unknown_limit_cannot_trade(self):
        inputs = fixture([["A"], ["B"]])
        inputs["signals"].loc[inputs["signals"].code.eq("B"), "group"] = 1
        inputs["portfolios"] = ["Q10"]
        result = run_execution(**inputs)
        second = result["weekly"].iloc[1]
        self.assertTrue(second.target_empty)
        self.assertEqual(second.cash_weight, 1.)
        self.assertEqual(second["return"], 0.)
        self.assertEqual(second.turnover, 1.)
        inputs = fixture([["A"]])
        inputs["status"]["limit_up"] = pd.Series(pd.NA, index=inputs["status"].index, dtype="boolean")
        result = run_execution(**inputs)
        self.assertEqual(result["trades"].rejection_reason.iloc[0], "unknown_limit_status")

    def test_missing_signal_week_preserves_shares_instead_of_liquidating(self):
        for missing_rows in (True, False):
            with self.subTest(missing_rows=missing_rows):
                inputs = fixture([["A"], [] if missing_rows else ["B"]])
                if not missing_rows:
                    inputs["signals"].loc[inputs["signals"].code.eq("B"), "group"] = np.nan
                result = run_execution(**inputs)
                second = result["weekly"].iloc[1]
                self.assertTrue(second.signal_missing)
                self.assertFalse(second.target_empty)
                self.assertEqual(second.turnover, 0.)
                self.assertEqual(second.cash_weight, 0.)
                self.assertEqual(second.n_stocks, 1)

    def test_calendar_and_reproducibility(self):
        inputs = fixture([["A"], ["B"]])
        first, second = run_execution(**inputs), run_execution(**inputs)
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])
        inputs["calendar"] = inputs["calendar"][1:]
        with self.assertRaises(DataGateError):
            run_execution(**inputs)


if __name__ == "__main__":
    unittest.main()
