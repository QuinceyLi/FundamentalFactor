"""AIQ adapter contracts: dates, event timing, units and explicit bad updates."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.aiq_market_cap import prepare_aiq_market_cap
from fundamental_factor.data import asof_snapshot


class AIQMarketCapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.calendar = self.write("calendar", pd.DataFrame({"date": [20250102, 20250103, 20250106, 20250107]}))

    def write(self, name, frame):
        path = self.root / f"{name}.feather"
        frame.to_feather(path)
        return path

    def run_adapter(self, frame, **kwargs):
        source = self.write("market_feature", frame)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = prepare_aiq_market_cap(source, self.calendar, self.root / "output", batch_size=2, **kwargs)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        data = pd.read_parquet(manifest["outputs"]["data"])
        quarantine = pd.read_parquet(manifest["outputs"]["quarantine"])
        return data, quarantine, manifest

    @staticmethod
    def caps(dates, values, codes=None):
        return pd.DataFrame({"date": dates, "code": codes or ["000001.SZ"] * len(dates),
                             "circulated_market_value": values, "turnover": [2.] * len(dates)})

    def test_calendar_domain_units_and_invalid_latest_record(self):
        data, quarantine, manifest = self.run_adapter(self.caps(
            [20250102, 20250103, 20250104, 20250106, 20250107, 20250102, 20250102],
            [50000., 0., 999999., -1., np.inf, 10., 20.],
            ["000001.SZ"] * 5 + ["000001.SH", "920001.BJ"]))
        valid = data[data.code.eq("000001.SZ")]
        self.assertEqual(valid.float_mcap.iloc[0], 500_000_000.)
        self.assertTrue(valid.float_mcap.iloc[1:].isna().all())
        self.assertTrue(data.code.eq("920001.BJ").any())
        self.assertIn("non_trading_date", quarantine.reason.values)
        self.assertIn("outside_a_share_code_domain", quarantine.reason.values)
        base = pd.DataFrame({"code": ["000001.SZ"], "cutoff": [pd.Timestamp("2025-01-05 23:59:59")]})
        snapshot = asof_snapshot(base, data, "mcap")
        self.assertTrue(snapshot.float_mcap.isna().all())
        self.assertEqual(snapshot.mcap_date.iloc[0], pd.Timestamp("2025-01-03"))
        self.assertEqual(manifest["counts"]["output_rows"], 5)

    def test_duplicates_across_batches_equal_and_conflicting(self):
        frame = self.caps([20250102, 20250103, 20250102, 20250103], [5., 10., 5., 11.])
        data, quarantine, manifest = self.run_adapter(frame)
        self.assertEqual(len(data), 2)
        self.assertEqual(data.float_mcap.iloc[0], 50_000.)
        self.assertTrue(np.isnan(data.float_mcap.iloc[1]))
        self.assertEqual(data.market_cap_status.iloc[1], "conflicting_duplicate")
        self.assertEqual(manifest["counts"]["duplicate_rows_removed"], 2)
        self.assertEqual(manifest["counts"]["conflicting_keys"], 1)
        self.assertTrue(quarantine.reason.eq("conflicting_duplicate").any())

    def test_share_event_waits_for_all_dates_and_invalid_event_blocks_old_value(self):
        shares = self.write("shares", pd.DataFrame({
            "date": [20250101, 20250103, 20250106], "reportdate": [20250101, 20250103, 20250106],
            "divdate": [20250101, 20250106, 20250107], "listdate": [20250101, 20250106, 20250107],
            "code": ["000001.SZ"] * 3, "float_shares": [5000., 6000., np.nan]}))
        quotes = self.write("quotes", pd.DataFrame({"date": [20250102, 20250103, 20250106, 20250107],
            "code": ["000001.SZ"] * 4, "close": [10.] * 4}))
        data, _, _ = self.run_adapter(self.caps([20250102, 20250103, 20250106, 20250107],
            [50000., 50000., 60000., 60000.]), share_source=shares, quote_source=quotes)
        self.assertEqual(data.share_implied_mcap.tolist()[:3], [500_000_000., 500_000_000., 600_000_000.])
        self.assertTrue(np.isnan(data.share_implied_mcap.iloc[3]))
        self.assertEqual(data.share_date.iloc[3], pd.Timestamp("2025-01-07"))
        self.assertTrue(data.float_mcap.notna().all())

    def test_same_day_proxy_diagnostic_and_corroborated_extreme_scale(self):
        quotes = self.write("quotes", pd.DataFrame({"date": [20250102, 20250103, 20250106],
            "code": ["000001.SZ"] * 3, "close": [10.] * 3, "volume": [10000.] * 3}))
        shares = self.write("shares", pd.DataFrame({"date": [20250101], "code": ["000001.SZ"],
                                                   "float_shares": [5000.]}))
        turnovers = self.write("turnover", pd.DataFrame({"date": [20250102, 20250103, 20250106],
            "code": ["000001.SZ"] * 3, "turnover": [2.] * 3}))
        data, quarantine, manifest = self.run_adapter(self.caps([20250102, 20250103, 20250106],
            [50000., 55000., 5_000_000.]), share_source=shares, quote_source=quotes,
            proxy_turnover_source=turnovers)
        self.assertEqual(data.proxy_mcap.tolist(), [500_000_000.] * 3)
        self.assertEqual(data.market_cap_crosscheck_status.iloc[0], "consistent_with_available_checks")
        self.assertEqual(data.float_mcap.iloc[1], 550_000_000.)
        self.assertEqual(data.market_cap_crosscheck_status.iloc[1], "diagnostic_disagreement")
        self.assertTrue(np.isnan(data.float_mcap.iloc[2]))
        self.assertEqual(data.market_cap_status.iloc[2], "extreme_magnitude_conflict")
        self.assertIn("diagnostic_disagreement_not_rejected", quarantine.reason.values)
        self.assertEqual(manifest["comparison_totals"]["share_compared"], 3)

    def test_no_asof_fill_of_missing_raw_quote_or_proxy_turnover(self):
        quotes = self.write("quotes", pd.DataFrame({"date": [20250102, 20250103],
            "code": ["000001.SZ"] * 2, "close": [10., np.nan], "volume": [10000.] * 2}))
        turnover = self.write("turnover", pd.DataFrame({"date": [20250102], "code": ["000001.SZ"],
                                                       "turnover": [2.]}))
        data, _, _ = self.run_adapter(self.caps([20250102, 20250103, 20250106], [50000.] * 3),
                                      quote_source=quotes, proxy_turnover_source=turnover)
        self.assertTrue(data.proxy_mcap.iloc[1:].isna().all())
        self.assertTrue(data.float_mcap.notna().all())

    def test_conflicting_share_and_quote_events_not_arbitrarily_selected(self):
        shares = self.write("shares", pd.DataFrame({"date": [20250101, 20250103, 20250103],
            "code": ["000001.SZ"] * 3, "float_shares": [5000., 6000., 7000.]}))
        quotes = self.write("quotes", pd.DataFrame({"date": [20250102, 20250102, 20250103],
            "code": ["000001.SZ"] * 3, "close": [10., 11., 10.]}))
        data, _, manifest = self.run_adapter(self.caps([20250102, 20250103], [50000.] * 2),
                                              share_source=shares, quote_source=quotes)
        self.assertTrue(data.share_implied_mcap.isna().all())
        self.assertEqual(manifest["input_stats"]["shares"]["conflicting_keys"], 1)
        self.assertTrue(data.float_mcap.notna().all())

    def test_reproducibility_and_nonfinite_values(self):
        frame = self.caps([20250102, 20250103, 20250106], [50000., np.nan, 1e308])
        first, _, m1 = self.run_adapter(frame)
        second, _, m2 = self.run_adapter(frame)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(m1["counts"], m2["counts"])
        self.assertTrue(second.float_mcap.iloc[1:].isna().all())

    def test_raw_close_required_and_input_cannot_be_overwritten(self):
        source = self.write("market_feature", self.caps([20250102], [50000.]))
        quote = self.write("adjusted", pd.DataFrame({"date": [20250102], "code": ["000001.SZ"],
                                                   "acc_close": [1000.]}))
        with self.assertRaisesRegex(ValueError, "missing fields.*close"):
            prepare_aiq_market_cap(source, self.calendar, self.root / "out", quote_source=quote)
        source = self.root / "float_mcap.parquet"
        self.caps([20250102], [50000.]).to_parquet(source, index=False)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_aiq_market_cap(source, self.calendar, self.root)


if __name__ == "__main__":
    unittest.main()
