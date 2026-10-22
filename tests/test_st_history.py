import unittest

import pandas as pd

from fundamental_factor.st_history import (exchange_st_events, normalize_szse_names,
                                           st_from_names, vendor_daily_st_events)


class STHistoryTests(unittest.TestCase):
    def test_historical_prefixes_and_missing_names(self):
        values = pd.Series(["ST华新", "*ST 华新", "S*ST华新", "SST华新", "G*ST环球",
                            "G ST农化", "Ｇ＊ＳＴ环球", "国农科技", "S深发展", None, "-"])
        actual = st_from_names(values)
        self.assertEqual(actual.iloc[:9].tolist(), [True] * 7 + [False, False])
        self.assertTrue(actual.iloc[9:].isna().all())

    def test_event_does_not_backfill_before_name(self):
        raw = pd.DataFrame({"变更日期": ["2020-05-01", "2021-05-01"], "证券代码": ["000001"] * 2,
                            "变更前简称": ["正常", "ST测试"], "变更后简称": ["ST测试", "正常"]})
        normalized = normalize_szse_names(raw, "2022-01-01")
        events = exchange_st_events(normalized, "2022-01-01")
        self.assertEqual(events.date.min(), pd.Timestamp("2020-05-01"))
        self.assertEqual(events.is_st.iloc[:2].tolist(), [True, False])
        self.assertTrue(pd.isna(events.is_st.iloc[-1]))
        self.assertEqual(events.date.iloc[-1], pd.Timestamp("2022-01-02"))

    def test_broken_chain_rejected(self):
        raw = pd.DataFrame({"变更日期": ["2020-05-01", "2021-05-01"], "证券代码": ["000001"] * 2,
                            "变更前简称": ["正常", "中间事件缺失"], "变更后简称": ["ST测试", "正常"]})
        with self.assertRaisesRegex(ValueError, "discontinuous"):
            normalize_szse_names(raw, "2022-01-01")

    def test_duplicate_or_future_event_rejected(self):
        raw = pd.DataFrame({"变更日期": ["2020-05-01"], "证券代码": ["000001"],
                            "变更前简称": ["正常"], "变更后简称": ["ST测试"]})
        with self.assertRaises(ValueError):
            normalize_szse_names(raw, "2019-01-01")
        with self.assertRaises(ValueError):
            normalize_szse_names(pd.concat([raw, raw]), "2022-01-01")

    def test_vendor_missing_day_ends_known_state(self):
        calendar = pd.Series(pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]))
        raw = pd.DataFrame({"date": calendar.iloc[[0, 2]].to_list(), "code": ["000001.SZ"] * 2,
                            "is_st": [True, False], "source": ["BaoStock"] * 2})
        events = vendor_daily_st_events(raw, calendar)
        self.assertEqual(events.date.to_list(), calendar.to_list())
        self.assertEqual(events.is_st.iloc[[0, 2]].tolist(), [True, False])
        self.assertTrue(events.is_st.iloc[[1, 3]].isna().all())

    def test_vendor_state_expires_after_calendar_coverage(self):
        calendar = pd.Series(pd.to_datetime(["2021-01-04", "2021-01-05"]))
        raw = pd.DataFrame({"date": calendar, "code": ["000001.SZ"] * 2,
                            "is_st": [True, True], "source": ["BaoStock"] * 2})
        events = vendor_daily_st_events(raw, calendar)
        self.assertEqual(events.date.to_list(), [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-06")])
        self.assertTrue(pd.isna(events.is_st.iloc[-1]))


if __name__ == "__main__":
    unittest.main()
