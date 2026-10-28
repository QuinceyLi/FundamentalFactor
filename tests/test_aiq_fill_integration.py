"""AIQ matrix filling is explicit, causal and traceable in the test pipeline.

All inputs here are synthetic; neither AIQ files nor protected market prices
are imported, changed or downloaded by these acceptance tests.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import load_config
from fundamental_factor.aiq_status import prepare_aiq_matrix_status
from fundamental_factor.market_cap import historical_market_cap_snapshot
from fundamental_factor.panel import build_panel, make_labels
from fundamental_factor.report import report
from fundamental_factor.research import run_research
from tests.test_aiq_integration import aiq_fixture


def set_mode(config, mode, output_name):
    raw = json.loads(config.path.read_text())
    if mode is None:
        raw.pop("missing_values", None)
    else:
        raw["missing_values"] = {"method": mode}
    raw["output_dir"] = str(config.path.parent / output_name)
    path = config.path.parent / f"{output_name}.json"
    path.write_text(json.dumps(raw, ensure_ascii=False))
    return load_config(path)


def select_feature(frame, code="000010.SZ", entry="2015-01-12", variant="raw", factor="ROIC_new"):
    return frame.loc[frame.code.eq(code) & frame.entry.eq(entry) &
                     frame.variant.eq(variant) & frame.factor.eq(factor)]


def one_holding(code, *, entry="2015-01-05", exit="2015-01-12", delisting=None):
    return pd.DataFrame({"entry": [pd.Timestamp(entry)], "exit": [pd.Timestamp(exit)], "code": [code],
                         "delisting_date": [pd.Timestamp(delisting) if delisting else pd.NaT]})


class AIQFillConfigurationTests(unittest.TestCase):
    def test_omitted_mode_preserves_old_defaults_and_new_mode_changes_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = aiq_fixture(Path(tmp), variants=["raw"])
            legacy = set_mode(base, None, "legacy")
            filled = set_mode(base, "aiq_matrix", "aiq")
            self.assertFalse(legacy.aiq_fill_enabled)
            self.assertEqual(legacy.missing_values, {"method": "preserve"})
            self.assertNotIn("missing_values", legacy.raw)
            self.assertTrue(filled.aiq_fill_enabled)
            self.assertNotEqual(legacy.digest, filled.digest)
            raw = json.loads(base.path.read_text())
            raw["missing_values"] = {"method": "bfill"}
            bad = Path(tmp) / "invalid.json"
            bad.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_config(bad)


class AIQFillFactorIntegrationTests(unittest.TestCase):
    def test_factor_ffill_retains_original_missing_and_donor_date_without_future_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = aiq_fixture(Path(tmp), variants=["raw"])
            source = pd.read_feather(base.source("roic"))
            source.loc[source.code.eq("000010.SZ") & source.date.eq("2015-01-02"), "ROIC_new"] = 2.5
            source.loc[source.code.eq("000010.SZ") & source.date.between("2015-01-05", "2015-01-09"), "ROIC_new"] = np.nan
            source.loc[source.code.eq("000010.SZ") & source.date.eq("2015-01-12"), "ROIC_new"] = 777.
            source.loc[source.code.eq("000011.SZ") & source.date.le("2015-01-09"), "ROIC_new"] = np.nan
            source.loc[source.code.eq("000011.SZ") & source.date.eq("2015-01-12"), "ROIC_new"] = 999.
            source.to_feather(base.source("roic"))
            before = hashlib.sha256(base.source("roic").read_bytes()).hexdigest()
            legacy = set_mode(base, "preserve", "legacy")
            new = set_mode(base, "aiq_matrix", "aiq")
            old_manifest, new_manifest = build_panel(legacy), build_panel(new)
            old = pd.read_parquet(legacy.output / "features.parquet")
            current = pd.read_parquet(new.output / "features.parquet")
            self.assertTrue(select_feature(old).empty)
            selected = select_feature(current).iloc[0]
            self.assertEqual(selected.value, 2.5)
            self.assertEqual(selected.signal, 2.5)
            self.assertTrue(pd.isna(selected.value_original))
            self.assertTrue(selected.value_original_missing)
            self.assertTrue(selected.value_filled)
            self.assertEqual(selected.value_source_date, pd.Timestamp("2015-01-02"))
            self.assertEqual(selected.value_value_source_date, selected.value_source_date)
            self.assertLessEqual(selected.value_source_date, selected.cutoff)
            self.assertTrue(select_feature(current, code="000011.SZ").empty)
            self.assertNotEqual(old_manifest["calculation_version"], new_manifest["calculation_version"])
            coverage = pd.read_csv(new.output / "coverage.csv")
            row = coverage.loc[coverage.factor.eq("ROIC_new") & coverage.entry.eq("2015-01-12")].iloc[0]
            self.assertEqual(row.factor_original_missing, 2)
            self.assertEqual(row.factor_filled, 1)
            self.assertEqual(hashlib.sha256(base.source("roic").read_bytes()).hexdigest(), before)
            run_research(new)
            report_path = report(new.output)
            self.assertTrue(report_path.is_file())
            frozen = json.loads((new.output / "config.frozen.json").read_text())
            self.assertEqual(frozen["missing_values"]["method"], "aiq_matrix")

    def test_factor_source_field_selects_zero_or_keep_instead_of_unconditional_ffill(self):
        for field, expected in (("volume", 0.), ("fy1", None)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                base = aiq_fixture(Path(tmp), variants=["raw"])
                source = pd.read_feather(base.source("roic")).rename(columns={"ROIC_new": field})
                source.loc[source.code.eq("000010.SZ") & source.date.eq("2015-01-02"), field] = 12.
                source.loc[source.code.eq("000010.SZ") & source.date.between("2015-01-05", "2015-01-09"), field] = np.nan
                source.loc[source.code.eq("000011.SZ") & source.date.le("2015-01-09"), field] = np.nan
                source.to_feather(base.source("roic"))
                raw = json.loads(base.path.read_text())
                raw["factors"] = [{"name": "TestField", "label": field, "source": "roic", "column": field}]
                base.path.write_text(json.dumps(raw))
                config = set_mode(load_config(base.path), "aiq_matrix", "filled")
                build_panel(config)
                features = pd.read_parquet(config.output / "features.parquet")
                chosen = select_feature(features, factor="TestField")
                if expected is None:
                    self.assertTrue(chosen.empty)
                else:
                    row = chosen.iloc[0]
                    self.assertEqual(row.value, expected)
                    self.assertTrue(row.value_filled)
                    self.assertTrue(pd.isna(row.value_original))
                    self.assertEqual(row.value_fill_anchor_date, pd.Timestamp("2015-01-02"))
                self.assertTrue(select_feature(features, code="000011.SZ", factor="TestField").empty)


class AIQFillControlsIntegrationTests(unittest.TestCase):
    def test_industry_matrix_continues_but_expired_st_event_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = aiq_fixture(Path(tmp), variants=["raw", "industry"])
            industry = pd.read_parquet(base.source("aiq_industry"))
            expired = industry.loc[industry.code.eq("000010.SZ")].copy()
            expired["date"] = pd.Timestamp("2015-01-09")
            expired["sector"] = None
            expired["industry_mapping_status"] = "interval_expired"
            pd.concat([industry, expired], ignore_index=True).to_parquet(base.source("aiq_industry"), index=False)
            st = pd.read_parquet(base.source("history"))
            st["is_st"] = st.is_st.astype("boolean")
            st["st_status_reason"] = "synthetic_known_state"
            expired_st = st.loc[st.code.eq("000010.SZ")].copy()
            expired_st["date"] = pd.Timestamp("2015-01-09")
            expired_st["is_st"] = pd.NA
            expired_st["st_status_reason"] = "coverage_expiry"
            pd.concat([st, expired_st], ignore_index=True).to_parquet(base.source("history"), index=False)
            old, new = set_mode(base, "preserve", "legacy"), set_mode(base, "aiq_matrix", "aiq")
            build_panel(old)
            build_panel(new)
            self.assertTrue(select_feature(pd.read_parquet(old.output / "features.parquet"), variant="industry").empty)
            controls = pd.read_parquet(new.output / "controls.parquet")
            row = controls.loc[controls.code.eq("000010.SZ") & controls.entry.eq("2015-01-12")].iloc[0]
            self.assertEqual(row.industry, "计算机")
            self.assertTrue(pd.isna(row.industry_original))
            self.assertTrue(row.industry_filled)
            self.assertEqual(row.industry_value_source_date, pd.Timestamp("2015-01-02"))
            self.assertEqual(row.industry_mapping_status, "interval_expired")
            self.assertTrue(pd.isna(row.is_st))
            self.assertTrue(row.is_st_unknown)
            self.assertEqual(row.st_status_reason, "coverage_expiry")

    def test_cap_nan_ffills_but_adapter_zero_negative_and_infinity_are_not_laundered(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = aiq_fixture(Path(tmp), variants=["raw", "size"])
            raw = json.loads(base.path.read_text())
            raw["neutralization"]["market_cap"]["column"] = "float_mcap"
            base.path.write_text(json.dumps(raw))
            base = load_config(base.path)
            codes = [f"{i:06d}.SZ" for i in range(10, 14)]
            previous = pd.DataFrame({"date": pd.Timestamp("2015-01-02"), "code": codes,
                                     "float_mcap": 1e8, "raw_mcap_wan": 10000.})
            latest = pd.DataFrame({"date": pd.Timestamp("2015-01-09"), "code": codes,
                                   "float_mcap": np.nan, "raw_mcap_wan": [np.nan, 0., -1., np.inf]})
            pd.concat([previous, latest], ignore_index=True).to_parquet(base.source("aiq_cap"), index=False)
            requests = pd.DataFrame({"cutoff": pd.Timestamp("2015-01-11 23:59"), "code": codes})
            preserved = historical_market_cap_snapshot(set_mode(base, "preserve", "legacy"), requests)
            self.assertTrue(preserved.float_mcap.isna().all())
            filled = historical_market_cap_snapshot(set_mode(base, "aiq_matrix", "aiq"), requests).set_index("code")
            row = filled.loc["000010.SZ"]
            self.assertEqual(row.float_mcap, 1e8)
            self.assertTrue(row.float_mcap_filled)
            self.assertTrue(pd.isna(row.float_mcap_original))
            self.assertEqual(row.float_mcap_value_source_date, pd.Timestamp("2015-01-02"))
            self.assertTrue(filled.loc[codes[1:], "float_mcap"].isna().all())
            self.assertFalse(filled.loc[codes[1:], "float_mcap_filled"].any())


class AIQFillReturnIntegrationTests(unittest.TestCase):
    def test_missing_adjusted_open_uses_previous_open_and_records_original_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = aiq_fixture(Path(tmp), variants=["raw"])
            quotes = pd.read_feather(base.source("quotes"))
            code = quotes.code.eq("000010.SZ")
            quotes.loc[code & quotes.date.eq("2015-01-09"), ["acc_open", "acc_close"]] = [110., 120.]
            quotes.loc[code & quotes.date.eq("2015-01-12"), "acc_open"] = np.nan
            quotes.to_feather(base.source("quotes"))
            before = hashlib.sha256(base.source("quotes").read_bytes()).hexdigest()
            holding = one_holding("000010.SZ")
            old = make_labels(set_mode(base, "preserve", "legacy"), holding).iloc[0]
            new = make_labels(set_mode(base, "aiq_matrix", "aiq"), holding).iloc[0]
            self.assertTrue(old.unresolved)
            self.assertFalse(new.unresolved)
            self.assertEqual(new.exit_level, 110.)
            self.assertAlmostEqual(new["return"], .1)
            self.assertTrue(new.exit_price_original_missing)
            self.assertTrue(new.exit_price_filled)
            self.assertEqual(new.exit_price_source_date, pd.Timestamp("2015-01-09"))
            self.assertFalse(new.entry_price_filled)
            self.assertEqual(hashlib.sha256(base.source("quotes").read_bytes()).hexdigest(), before)

    def test_first_observation_missing_and_invalid_zero_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = aiq_fixture(Path(tmp), variants=["raw"])
            quotes = pd.read_feather(base.source("quotes"))
            quotes.loc[quotes.code.eq("000010.SZ") & quotes.date.le("2015-01-05"), "acc_open"] = np.nan
            quotes.loc[quotes.code.eq("000010.SZ") & quotes.date.eq("2015-01-06"), "acc_open"] = 111.
            quotes.loc[quotes.code.eq("000011.SZ") & quotes.date.eq("2015-01-12"), "acc_open"] = 0.
            quotes.to_feather(base.source("quotes"))
            config = set_mode(base, "aiq_matrix", "aiq")
            first = make_labels(config, one_holding("000010.SZ")).iloc[0]
            zero = make_labels(config, one_holding("000011.SZ")).iloc[0]
            self.assertTrue(first.unresolved)
            self.assertTrue(pd.isna(first.entry_level))
            self.assertFalse(first.entry_price_filled)
            self.assertTrue(first.entry_price_original_missing)
            self.assertTrue(zero.unresolved)
            self.assertFalse(zero.exit_price_filled)

    def test_no_ffill_exit_value_after_delisting_without_actual_settlement(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = aiq_fixture(Path(tmp), variants=["raw"])
            quotes = pd.read_feather(base.source("quotes"))
            quotes.loc[quotes.code.eq("000010.SZ") & quotes.date.ge("2015-01-09"), ["acc_open", "acc_close"]] = np.nan
            quotes.to_feather(base.source("quotes"))
            config = set_mode(base, "aiq_matrix", "aiq")
            row = make_labels(config, one_holding("000010.SZ", delisting="2015-01-09")).iloc[0]
            self.assertTrue(row.unresolved)
            self.assertTrue(row.delisting_without_terminal)
            self.assertTrue(pd.isna(row["return"]))

    def test_prelisting_price_cannot_supply_postlisting_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = aiq_fixture(Path(tmp), variants=["raw"])
            stocks = pd.read_parquet(base.source("stocks"))
            stocks.loc[stocks.code.eq("000010.SZ"), "listing_date"] = pd.Timestamp("2015-01-12")
            stocks.to_parquet(base.source("stocks"), index=False)
            quotes = pd.read_feather(base.source("quotes"))
            quotes.loc[quotes.code.eq("000010.SZ") & quotes.date.ge("2015-01-12"), ["acc_open", "acc_close"]] = np.nan
            quotes.to_feather(base.source("quotes"))
            row = make_labels(set_mode(base, "aiq_matrix", "aiq"), one_holding(
                "000010.SZ", entry="2015-01-19", exit="2015-01-26")).iloc[0]
            self.assertTrue(row.unresolved)
            self.assertTrue(pd.isna(row.entry_level))
            self.assertTrue(pd.isna(row["return"]))


class AIQFilledStatusIntegrationTests(unittest.TestCase):
    def test_daily_grid_zero_volume_proxies_preserve_lifecycle_and_cross_year_donors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calendar = pd.to_datetime(["2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06"])
            pd.DataFrame({"date": calendar}).to_parquet(root / "calendar.parquet", index=False)
            stocks = pd.DataFrame({"code": ["600001.SH", "600002.SH", "600003.SH"],
                "listing_date": pd.to_datetime(["2010-01-01", "2020-01-02", "2010-01-01"]),
                "delisting_date": pd.to_datetime([None, None, "2020-01-03"])})
            stocks.to_parquet(root / "stocks.parquet", index=False)
            quotes = pd.DataFrame({"date": pd.to_datetime(["2019-12-30", "2020-01-03", "2019-12-30", "2020-01-03", "2019-12-30"]),
                "code": ["600001.SH", "600001.SH", "600002.SH", "600002.SH", "600003.SH"],
                "open": [10., 11., 20., 21., 30.], "close": [10., 11., 20., 21., 30.],
                "last": [9., 10., 19., 20., 29.], "volume": [100., np.nan, 200., 50., 300.],
                "acc_close": [999999.] * 5})
            quotes.to_feather(root / "quotes.feather")
            st = pd.DataFrame({"date": pd.to_datetime(["2019-12-30", "2019-12-30", "2019-12-30", "2020-01-02"]),
                               "code": ["600001.SH", "600002.SH", "600003.SH", "600001.SH"],
                               "is_st": pd.array([False, False, False, pd.NA], dtype="boolean")})
            st.to_parquet(root / "st.parquet", index=False)
            digest = hashlib.sha256((root / "quotes.feather").read_bytes()).hexdigest()
            args = [root / "quotes.feather", root / "calendar.parquet", root / "stocks.parquet", root / "st.parquet"]
            summary = prepare_aiq_matrix_status(*args, root / "one.parquet", start=calendar[0], end=calendar[-1], stock_batch_size=1)
            prepare_aiq_matrix_status(*args, root / "all.parquet", start=calendar[0], end=calendar[-1], stock_batch_size=3)
            def sorted_output(path):
                return pd.read_parquet(path).sort_values(["date", "code"]).reset_index(drop=True)
            result = sorted_output(root / "one.parquet")
            pd.testing.assert_frame_equal(result, sorted_output(root / "all.parquet"), check_exact=True)
            self.assertEqual(len(result), 11)
            self.assertNotIn("acc_close", result)
            gap = result[result.code.eq("600001.SH") & result.date.eq("2020-01-02")].iloc[0]
            self.assertTrue(gap.suspended)
            self.assertTrue(gap.st_unknown)
            self.assertEqual(gap.volume, 0.)
            self.assertTrue(gap.volume_original_missing)
            self.assertTrue(gap.volume_filled)
            self.assertTrue(pd.isna(gap.volume_value_source_date))
            self.assertEqual(gap.volume_fill_anchor_date, pd.Timestamp("2019-12-30"))
            self.assertEqual(gap.close, 10.)
            self.assertEqual(gap.close_value_source_date, pd.Timestamp("2019-12-30"))
            self.assertTrue(pd.isna(gap.open))
            self.assertFalse(gap.open_filled)
            first = result[result.code.eq("600002.SH") & result.date.eq("2020-01-02")].iloc[0]
            self.assertTrue(pd.isna(first.suspended))
            self.assertTrue(pd.isna(first.volume))
            self.assertTrue(pd.isna(first.close))
            self.assertFalse(first.close_filled)
            self.assertFalse((result.code.eq("600003.SH") & result.date.ge("2020-01-03")).any())
            self.assertEqual(summary["source_outside_lifecycle_rows_ignored"], 1)
            self.assertGreater(summary["counts"]["volume_filled_rows"], 0)
            self.assertEqual(hashlib.sha256((root / "quotes.feather").read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
