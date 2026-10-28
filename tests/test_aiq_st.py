import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fundamental_factor.aiq_st import build_aiq_st_events, prepare_aiq_st


def events(rows):
    return pd.DataFrame(rows, columns=["code", "execute_date", "cancel_date", "announce_date"])


def names(rows):
    return pd.DataFrame(rows, columns=["code", "execute_date", "cancel_date", "name"])


def state_at(frame, date, code="600001.SH"):
    rows = frame.loc[frame.code.eq(code) & frame.date.le(pd.Timestamp(date))]
    return rows.iloc[-1] if len(rows) else None


class AIQSTTests(unittest.TestCase):
    def fixtures(self):
        return (events([("600001.SH", 20200104, 20200108, 20200107),
                        ("000002.SZ", 20200120, None, 20200119)]),
                names([("600001.SH", 20200101, 20200103, "正常公司"),
                       ("600001.SH", 20200104, 20200107, "*ST公司"),
                       ("600001.SH", 20200108, None, "正常公司"),
                       ("000002.SZ", 20200120, None, "ST另股")]))

    def test_announcement_after_execute_does_not_delay_historical_state(self):
        ev, nm = self.fixtures()
        result, quarantine, _, meta = build_aiq_st_events(ev, nm)
        self.assertTrue(state_at(result, "2020-01-04").is_st)
        self.assertEqual(state_at(result, "2020-01-04").st_status_reason, "st_event_and_name_agree")
        self.assertEqual(state_at(result, "2020-01-04").st_announce_dates, "2020-01-07")
        self.assertEqual(meta["announcement_after_execute_rows"], 1)
        self.assertIn("announcement_after_effective_date_semantics_unverified", quarantine.reason.tolist())

    def test_name_end_inclusive_st_cancel_exclusive_and_no_early_backfill(self):
        ev, nm = self.fixtures()
        result, _, _, _ = build_aiq_st_events(ev, nm)
        self.assertIsNone(state_at(result, "2019-12-31"))
        self.assertFalse(state_at(result, "2020-01-03").is_st)
        self.assertTrue(state_at(result, "2020-01-07").is_st)
        self.assertFalse(state_at(result, "2020-01-08").is_st)
        self.assertIsNone(state_at(result, "2020-01-08", "600999.SH"))

    def test_old_open_name_expires_even_when_event_source_is_newer(self):
        ev, nm = self.fixtures()
        ev = pd.concat([ev, events([("600001.SH", 20230102, None, 20230101)])], ignore_index=True)
        result, _, _, meta = build_aiq_st_events(ev, nm)
        self.assertEqual(meta["name_coverage_through"], pd.Timestamp("2020-01-20"))
        self.assertFalse(state_at(result, "2020-01-20").is_st)
        self.assertTrue(pd.isna(state_at(result, "2020-01-21").is_st))
        self.assertTrue(pd.isna(state_at(result, "2023-01-02").is_st))
        self.assertEqual(state_at(result, "2023-01-02").st_status_reason, "name_coverage_expired")
        self.assertTrue(pd.isna(state_at(result, "2023-01-03").is_st))

    def test_overlapping_events_do_not_cancel_each_other(self):
        ev, nm = self.fixtures()
        ev = pd.concat([ev, events([("600001.SH", 20200105, 20200106, 20200105)])], ignore_index=True)
        result, _, _, _ = build_aiq_st_events(ev, nm)
        self.assertTrue(state_at(result, "2020-01-06").is_st)
        self.assertEqual(state_at(result, "2020-01-06").st_status_reason, "st_event_and_name_agree")
        self.assertEqual(state_at(result, "2020-01-05").st_source_rows, "2;4")

    def test_retirement_and_conflicting_untyped_events_stay_unknown(self):
        ev = events([("600001.SH", 20200101, None, 20200101),
                     ("000002.SZ", 20200120, None, 20200119)])
        nm = names([("600001.SH", 20200101, 20200109, "退市公司"),
                    ("600001.SH", 20200110, None, "普通公司"),
                    ("000002.SZ", 20200120, None, "ST另股")])
        result, quarantine, _, _ = build_aiq_st_events(ev, nm)
        self.assertTrue(pd.isna(state_at(result, "2020-01-01").is_st))
        self.assertTrue(pd.isna(state_at(result, "2020-01-10").is_st))
        self.assertIn("untyped_event_conflicts_with_non_st_name", quarantine.reason.tolist())

    def test_name_gaps_emit_unknown_before_future_interval(self):
        ev, nm = self.fixtures()
        nm.loc[1, "cancel_date"] = 20200105
        result, _, _, _ = build_aiq_st_events(ev, nm)
        self.assertTrue(pd.isna(state_at(result, "2020-01-06").is_st))
        self.assertFalse(state_at(result, "2020-01-08").is_st)

    def test_names_without_matching_event_remain_explicit_proxies(self):
        ev, nm = self.fixtures()
        ev = ev.iloc[[1]].reset_index(drop=True)
        result, _, _, _ = build_aiq_st_events(ev, nm)
        row = state_at(result, "2020-01-04")
        self.assertTrue(row.is_st)
        self.assertEqual(row.st_status_reason, "st_name_only_proxy")
        self.assertEqual(row.st_evidence_priority, 2)
        self.assertTrue(row.is_proxy)

    def test_exact_duplicates_collapse_but_conflicting_names_do_not_choose_arbitrarily(self):
        ev, nm = self.fixtures()
        ev = pd.concat([ev, ev.iloc[[0]]], ignore_index=True)
        nm = pd.concat([nm, names([("600001.SH", 20200105, 20200106, "普通公司")])], ignore_index=True)
        result, quarantine, _, meta = build_aiq_st_events(ev, nm)
        self.assertEqual(meta["event_rows_after_exact_deduplication"], 2)
        self.assertFalse(result.duplicated(["date", "code"]).any())
        self.assertTrue(pd.isna(state_at(result, "2020-01-05").is_st))
        self.assertIn("exact_duplicate_collapsed", quarantine.reason.tolist())
        self.assertIn("conflicting_name_intervals", quarantine.reason.tolist())

    def test_invalid_interval_cannot_create_non_st_clearance(self):
        ev, nm = self.fixtures()
        ev.loc[0, "cancel_date"] = 20190101
        result, quarantine, _, _ = build_aiq_st_events(ev, nm)
        self.assertTrue(pd.isna(state_at(result, "2020-01-08").is_st))
        self.assertIn("invalid_code_or_interval", quarantine.reason.tolist())

    def test_coverage_cannot_be_extended_beyond_source_evidence(self):
        ev, nm = self.fixtures()
        with self.assertRaisesRegex(ValueError, "cannot extend"):
            build_aiq_st_events(ev, nm, names_valid_through="2026-01-01")

    def test_export_has_audit_sidecars_preserves_weekends_and_is_reproducible(self):
        ev, nm = self.fixtures()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input"
            inputs.mkdir()
            ev.to_csv(inputs / "st.csv", index=False)
            nm.to_excel(inputs / "names.xlsx", index=False)
            pd.DataFrame({"date": [20200103, 20200106, 20200107, 20200108, 20200120]}).to_csv(
                inputs / "calendar.csv", index=False)
            output = root / "output" / "st.parquet"
            summary = prepare_aiq_st(inputs / "st.csv", inputs / "names.xlsx", output,
                                     calendar_source=inputs / "calendar.csv")
            first = pd.read_parquet(output)
            self.assertEqual(state_at(first, "2020-01-04").date, pd.Timestamp("2020-01-04"))
            self.assertGreater(summary["calendar"]["non_session_event_boundaries_retained"], 0)
            self.assertTrue(output.with_suffix(".manifest.json").exists())
            self.assertTrue(output.with_suffix(".coverage.parquet").exists())
            self.assertTrue(output.with_suffix(".quarantine.csv").exists())
            prepare_aiq_st(inputs / "st.csv", inputs / "names.xlsx", output)
            pd.testing.assert_frame_equal(first, pd.read_parquet(output))
            with self.assertRaisesRegex(ValueError, "read-only"):
                prepare_aiq_st(inputs / "st.csv", inputs / "names.xlsx", inputs / "derived.parquet")


if __name__ == "__main__":
    unittest.main()
