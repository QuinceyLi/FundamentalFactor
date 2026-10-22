import unittest
import pandas as pd
from fundamental_factor.proxy_inputs import history_from_sources, trade_status_proxy


class ProxyInputTests(unittest.TestCase):
    def test_future_st_does_not_rewrite_history_and_null_reset_is_preserved(self):
        keys = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
                             "code": ["000001.SZ"] * 3})
        classes = keys.assign(**{"class": [1., 2., 35.]})
        st = keys.iloc[1:].assign(is_st=pd.array([True, pd.NA], dtype="boolean"))
        result = history_from_sources(keys, classes, st)
        self.assertTrue(pd.isna(result.is_st.iloc[0]))
        self.assertTrue(result.is_st.iloc[1])
        self.assertTrue(pd.isna(result.is_st.iloc[2]))
        self.assertEqual(result.industry.tolist(), ["class_1", "class_2", "class_35"])

    def test_reference_price_direction_st_regime_and_unknowns(self):
        quotes = pd.DataFrame({"date": pd.to_datetime(["2020-08-21", "2020-08-24", "2020-08-24", "2020-08-24", "2020-08-24"]),
            "code": ["300001.SZ", "300001.SZ", "000001.SZ", "600001.SH", "600002.SH"],
            "open": [11., 11., 10.5, 9.5, 11.], "close": [11., 11., 10.5, 9.5, 11.],
            "last": [10.] * 5, "volume": [1., 1., 1., 0., 1.]})
        history = quotes[["date", "code"]].assign(is_st=pd.array([False, False, True, True, pd.NA], dtype="boolean"))
        stocks = pd.DataFrame({"code": ["300001.SZ", "000001.SZ", "600001.SH", "600002.SH"],
            "listing_date": pd.to_datetime(["2010-01-01"] * 3 + ["2020-08-24"])})
        out = trade_status_proxy(quotes, history, stocks, pd.bdate_range("2010-01-01", "2020-08-24"))
        self.assertTrue(out.limit_up.iloc[0])
        self.assertFalse(out.limit_up.iloc[1])
        self.assertTrue(out.limit_up.iloc[2])
        self.assertTrue(out.limit_down.iloc[3])
        self.assertTrue(out.suspended.iloc[3])
        self.assertTrue(pd.isna(out.limit_up.iloc[4]))
        # The original flag remains unknown even though a board proxy can use 10%.
        self.assertTrue(out.st_unknown.iloc[4])

    def test_nonconsecutive_input_index_does_not_realign_stock_flags(self):
        quotes = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 2),
            "code": ["000001.SZ", "000002.SZ"], "open": [11., 9.],
            "close": [11., 9.], "last": [10., 10.], "volume": [1., 1.]}, index=[10, 20])
        history = quotes[["date", "code"]].assign(is_st=False)
        stocks = pd.DataFrame({"code": quotes.code.tolist(),
                               "listing_date": pd.to_datetime(["2010-01-01"] * 2)})
        out = trade_status_proxy(quotes, history, stocks, pd.bdate_range("2010-01-01", "2024-01-02"))
        self.assertEqual(out.code.tolist(), quotes.code.tolist())
        self.assertEqual(out.limit_up.tolist(), [True, False])
        self.assertEqual(out.limit_down.tolist(), [False, True])

    def test_limit_rounding_half_cent_does_not_trigger_one_cent_early(self):
        quotes = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 2),
            "code": ["000001.SZ", "000002.SZ"], "open": [4.51, 1.26],
            "close": [4.51, 1.26], "last": [4.30, 1.15], "volume": [1., 1.]})
        history = quotes[["date", "code"]].assign(is_st=[True, False])
        stocks = pd.DataFrame({"code": quotes.code.tolist(),
                               "listing_date": pd.to_datetime(["2010-01-01"] * 2)})
        out = trade_status_proxy(quotes, history, stocks, pd.bdate_range("2010-01-01", "2024-01-02"))
        # 4.30 * 1.05 = 4.515 -> 4.52; 1.15 * 1.10 = 1.265 -> 1.27.
        self.assertEqual(out.limit_up.tolist(), [False, False])

    def test_main_board_st_limit_regime_changes_on_20260706(self):
        quotes = pd.DataFrame({"date": pd.to_datetime(["2026-07-03", "2026-07-06"] * 2),
            "code": ["000001.SZ"] * 2 + ["600001.SH"] * 2,
            "open": [10.5] * 4, "close": [10.5] * 4,
            "last": [10.] * 4, "volume": [1.] * 4})
        history = quotes[["date", "code"]].assign(is_st=True)
        stocks = pd.DataFrame({"code": ["000001.SZ", "600001.SH"],
                               "listing_date": pd.to_datetime(["2010-01-01"] * 2)})
        out = trade_status_proxy(quotes, history, stocks, pd.bdate_range("2010-01-01", "2026-07-06"))
        self.assertEqual(out.limit_up.tolist(), [True, False, True, False])
        self.assertEqual(out.limit_pct_proxy.tolist(), [.05, .10, .05, .10])
