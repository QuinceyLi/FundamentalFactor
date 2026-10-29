"""Synthetic acceptance of AIQ inputs, source precedence and cutoff semantics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.aiq_inputs import merge_st_sources, prepare_stocks
from fundamental_factor.audit import source_columns
from fundamental_factor.config import load_config
from fundamental_factor.market_cap import historical_market_cap_snapshot
from fundamental_factor.panel import _industry_controls, build_panel
from fundamental_factor.report import report
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture


def st_events(rows, source=None):
    frame = pd.DataFrame(rows, columns=["date", "code", "is_st"])
    frame["date"] = pd.to_datetime(frame.date)
    frame["is_st"] = frame.is_st.astype("boolean")
    if source:
        frame["st_source"] = source
        frame["st_status_reason"] = "synthetic_evidence"
    return frame


def state_at(frame, date, code="600001.SH"):
    rows = frame[frame.code.eq(code) & frame.date.le(pd.Timestamp(date))].sort_values("date")
    return rows.iloc[-1] if len(rows) else None


def aiq_fixture(root, variants=None):
    config = disk_fixture(root, weeks_count=6, stocks_count=100)
    data = Path(config.raw["data_root"])
    old = pd.read_parquet(config.source("history"))
    # The ST source deliberately has neither industry nor capitalisation.
    old[["date", "code", "is_st"]].to_parquet(config.source("history"), index=False)
    industry = old[["date", "code"]].copy()
    industry["sector"] = np.resize(["AIQ_ZX_25", "AIQ_ZX_24", "AIQ_ZX_33", "AIQ_ZX_01"], len(old))
    industry["custom_class_code"] = np.resize([25, 24, 33, 1], len(old))
    industry["citic_level1_code"] = np.resize([27, 35, 62, 40], len(old))
    industry["citic_level1_name"] = np.resize(["电力设备及新能源", "医药", "计算机", "银行"], len(old))
    industry["industry_mapping_status"] = "mapped"
    industry.to_parquet(data / "aiq_industry.parquet", index=False)
    cap = old[["date", "code", "float_mcap"]].rename(columns={"float_mcap": "float_value_yuan"})
    cap["market_cap_source"] = "synthetic_aiq_market_feature"
    cap["market_cap_status"] = "valid"
    cap["market_cap_crosscheck_status"] = "synthetic_crosscheck"
    cap.to_parquet(data / "aiq_cap.parquet", index=False)
    raw = json.loads(config.path.read_text())
    raw["data_policy"] = {"mode": "classic", "missing_returns": "available_case", "empty_groups": "cash"}
    raw["preprocess"] = {"winsorize": False, "standardize": False}
    raw["industry"] = {"classification": "aiq_zx35", "source": "aiq_industry", "column": "sector"}
    raw["neutralization"] = {"variants": variants or ["raw", "industry", "size", "industry_size", "full"],
                             "sample_policy": "independent", "market_cap": {
                                 "method": "history", "source": "aiq_cap", "column": "float_value_yuan"}}
    raw["sources"].update(aiq_industry="aiq_industry.parquet", aiq_cap="aiq_cap.parquet")
    # The declaration enables audit exports; these are all synthetic inputs.
    manifest = data / "inputs_manifest.json"
    manifest.write_text(json.dumps({"dataset_kind": "synthetic", "description": "AIQ integration fixture"}))
    raw["input_manifest"] = str(manifest)
    config.path.write_text(json.dumps(raw, ensure_ascii=False))
    return load_config(config.path)


class AIQSTMergeTests(unittest.TestCase):
    def test_aiq_agreement_unknown_fallback_and_both_source_expiry(self):
        aiq = st_events([("2020-01-01", "600001.SH", True), ("2020-01-03", "600001.SH", None),
                         ("2020-01-05", "600001.SH", False), ("2020-01-07", "600001.SH", None)], "AIQ_dual")
        official = st_events([("2020-01-02", "600001.SH", True),
                              ("2020-01-04", "600001.SH", False), ("2020-01-06", "600001.SH", None)])
        merged, conflicts = merge_st_sources(aiq, official)
        self.assertTrue(conflicts.empty)
        self.assertEqual(state_at(merged, "2020-01-02").st_source, "AIQ_dual")
        self.assertTrue(state_at(merged, "2020-01-03").is_st)
        self.assertEqual(state_at(merged, "2020-01-03").st_status_reason, "official_fallback")
        self.assertFalse(state_at(merged, "2020-01-04").is_st)
        self.assertEqual(state_at(merged, "2020-01-05").st_source, "AIQ_dual")
        self.assertFalse(state_at(merged, "2020-01-06").is_st)
        self.assertTrue(pd.isna(state_at(merged, "2020-02-01").is_st))
        self.assertFalse(state_at(merged, "2020-02-01").st_known)
        self.assertIsNone(state_at(merged, "2019-12-31"))

    def test_official_conflict_wins_only_after_its_effective_date(self):
        aiq = st_events([("2020-01-01", "600001.SH", True),
                         ("2020-01-05", "600001.SH", False)], "AIQ_dual")
        official = st_events([("2020-01-02", "600001.SH", False),
                              ("2020-01-04", "600001.SH", True)])
        merged, conflicts = merge_st_sources(aiq, official)
        self.assertTrue(state_at(merged, "2020-01-01").is_st)
        self.assertFalse(state_at(merged, "2020-01-02").is_st)
        self.assertEqual(state_at(merged, "2020-01-02").st_status_reason,
                         "official_event_overrides_aiq_conflict")
        self.assertEqual(state_at(merged, "2020-01-04").st_source, "AIQ_dual")
        self.assertTrue(state_at(merged, "2020-01-05").is_st)
        self.assertEqual(conflicts.date.tolist(), list(pd.to_datetime(["2020-01-02", "2020-01-05"])))
        self.assertTrue(conflicts.st_conflict.all())
        self.assertFalse(merged.duplicated(["date", "code"]).any())

    def test_unseen_security_and_future_official_name_do_not_clear_past_unknown(self):
        aiq = st_events([("2020-01-01", "600001.SH", None),
                         ("2020-01-01", "000001.SZ", True)], "AIQ")
        official = st_events([("2020-01-04", "600001.SH", False)])
        merged, _ = merge_st_sources(aiq, official)
        self.assertTrue(pd.isna(state_at(merged, "2020-01-03").is_st))
        self.assertFalse(state_at(merged, "2020-01-04").is_st)
        self.assertTrue(state_at(merged, "2020-01-04", "000001.SZ").is_st)
        self.assertIsNone(state_at(merged, "2020-01-04", "600999.SH"))


class AIQControlsIntegrationTests(unittest.TestCase):
    def test_same_history_file_uses_explicit_cap_column_instead_of_old_float_mcap(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = aiq_fixture(Path(tmp), variants=["raw", "size"])
            history = pd.read_parquet(config.source("history"))
            expected = pd.read_parquet(config.source("aiq_cap"))[["date", "code", "float_value_yuan"]]
            history = history.merge(expected, on=["date", "code"], validate="one_to_one")
            history["float_mcap"] = 99.  # A stale, unrelated field must not win.
            history.to_parquet(config.source("history"), index=False)
            raw = json.loads(config.path.read_text())
            raw["neutralization"]["market_cap"]["source"] = "history"
            config.path.write_text(json.dumps(raw))
            config = load_config(config.path)
            self.assertTrue(config.separate_market_cap_source)
            self.assertIn("float_value_yuan", source_columns(config)["history"])
            self.assertNotIn("float_mcap", source_columns(config)["history"])
            build_panel(config)
            controls = pd.read_parquet(config.output / "controls.parquet")
            self.assertTrue(controls.float_mcap.gt(99.).all())
            self.assertTrue(controls.float_mcap.eq(controls.float_value_yuan).all())

    def test_independent_config_sources_are_resolved_without_polluting_st_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = aiq_fixture(Path(tmp))
            self.assertEqual(config.industry["classification"], "aiq_zx35")
            self.assertTrue(config.separate_industry_source)
            self.assertTrue(config.separate_market_cap_source)
            self.assertFalse(config.uses_market_cap_proxy)
            self.assertEqual(config.source("aiq_cap"), Path(tmp) / "data" / "aiq_cap.parquet")
            selected = source_columns(config)
            self.assertIn("float_value_yuan", selected["aiq_cap"])
            self.assertIn("sector", selected["aiq_industry"])
            self.assertNotIn("float_mcap", selected["history"])
            self.assertNotIn("industry", selected["history"])
            for invalid in ({"source": "calendar"}, {"column": "date"}):
                raw = json.loads(config.path.read_text())
                raw["neutralization"]["market_cap"].update(invalid)
                alternate = Path(tmp) / "bad_config.json"
                alternate.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    load_config(alternate)

    def test_latest_null_control_updates_block_older_values_and_future_rescue(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = aiq_fixture(Path(tmp))
            industry = pd.DataFrame({"date": pd.to_datetime(["2015-01-02", "2015-01-09", "2015-01-12"]),
                "code": ["000000.SZ"] * 3, "sector": ["AIQ_ZX_25", None, "AIQ_ZX_24"],
                "industry_mapping_status": ["mapped", "interval_expired", "mapped"]})
            industry.to_parquet(config.source("aiq_industry"), index=False)
            cap = pd.DataFrame({"date": industry.date, "code": industry.code,
                                "float_value_yuan": [1e8, 0., 9e9],
                                "market_cap_source": "synthetic_direct"})
            cap.to_parquet(config.source("aiq_cap"), index=False)
            base = pd.DataFrame({"cutoff": pd.to_datetime(["2015-01-08 23:59", "2015-01-11 23:59", "2015-01-12 23:59"]),
                                 "code": ["000000.SZ"] * 3, "industry": "stale_unrelated_history",
                                 "float_mcap": 99.})
            actual = historical_market_cap_snapshot(config, _industry_controls(config, base))
            self.assertEqual(actual.industry.tolist()[::2], ["太阳能及光伏", "化学制药"])
            self.assertTrue(pd.isna(actual.industry.iloc[1]))
            self.assertEqual(actual.industry_mapping_status.iloc[1], "interval_expired")
            self.assertEqual(actual.industry_source_date.iloc[1], pd.Timestamp("2015-01-09"))
            self.assertEqual(actual.float_mcap.iloc[0], 1e8)
            self.assertTrue(pd.isna(actual.float_mcap.iloc[1]))
            self.assertEqual(actual.market_cap_date.iloc[1], pd.Timestamp("2015-01-09"))
            self.assertEqual(actual.market_cap_invalid_reason.iloc[1], "invalid_historical_value")
            self.assertEqual(actual.float_mcap.iloc[2], 9e9)
            self.assertTrue(actual.market_cap_date.le(actual.cutoff).all())
            self.assertTrue(actual.industry_source_date.le(actual.cutoff).all())

    def test_invalid_cap_variants_remain_missing_instead_of_reusing_valid_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = aiq_fixture(Path(tmp))
            codes = [f"{i:06d}.SZ" for i in range(4)]
            initial = pd.DataFrame({"date": pd.Timestamp("2015-01-02"), "code": codes, "float_value_yuan": 1e8})
            latest = pd.DataFrame({"date": pd.Timestamp("2015-01-09"), "code": codes,
                                   "float_value_yuan": [np.nan, np.inf, -10., 0.]})
            pd.concat([initial, latest], ignore_index=True).to_parquet(config.source("aiq_cap"), index=False)
            base = pd.DataFrame({"cutoff": pd.Timestamp("2015-01-11"), "code": codes})
            actual = historical_market_cap_snapshot(config, base)
            self.assertTrue(actual.float_mcap.isna().all())
            self.assertTrue(actual.market_cap_date.eq(pd.Timestamp("2015-01-09")).all())

    def test_six_week_pipeline_raw_returns_survive_missing_industry_cap_and_barra(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = aiq_fixture(root, variants=["raw"])
            result = run_research(baseline)
            expected_features = pd.read_parquet(baseline.output / "features.parquet")
            expected_returns = {factor["name"]: pd.read_csv(baseline.output / factor["output_dir"] / "portfolio_weekly.csv")
                                for factor in result["factors"]}
            industry = pd.read_parquet(baseline.source("aiq_industry"))
            industry = industry[~industry.code.isin([f"{i:06d}.SZ" for i in range(5)])]
            industry.to_parquet(baseline.source("aiq_industry"), index=False)
            cap = pd.read_parquet(baseline.source("aiq_cap"))
            cap.loc[cap.code.isin([f"{i:06d}.SZ" for i in range(5, 10)]), "float_value_yuan"] = np.nan
            cap.to_parquet(baseline.source("aiq_cap"), index=False)
            raw = json.loads(baseline.path.read_text())
            raw["neutralization"]["variants"] = ["raw", "industry", "size", "industry_size", "full"]
            raw["sources"]["risk"] = None
            raw["output_dir"] = str(root / "with_missing_controls")
            baseline.path.write_text(json.dumps(raw))
            config = load_config(baseline.path)
            manifest = build_panel(config)
            self.assertEqual(manifest["periods"], 6)
            self.assertEqual(manifest["base_rows"], 600)
            self.assertEqual(manifest["calculation_version"], 7)
            features = pd.read_parquet(config.output / "features.parquet")
            columns = ["factor", "entry", "code", "value", "signal", "percentile", "group"]
            def raw_signals(frame):
                return frame.loc[frame.variant.eq("raw"), columns].sort_values(columns[:3]).reset_index(drop=True)
            pd.testing.assert_frame_equal(raw_signals(expected_features), raw_signals(features), check_exact=True)
            coverage = pd.read_csv(config.output / "coverage.csv")
            for variant, count in (("raw", 100), ("industry", 95), ("size", 95), ("industry_size", 90), ("full", 0)):
                self.assertTrue(coverage[f"variant_{variant}_count"].eq(count).all(), variant)
            actual_result = run_research(config)
            for factor in actual_result["factors"]:
                actual = pd.read_csv(config.output / factor["output_dir"] / "portfolio_weekly.csv")
                actual = actual[actual.variant.eq("raw")].reset_index(drop=True)
                # CSV infers all-empty text columns differently when other
                # variants contain diagnostics; numeric values remain exact.
                pd.testing.assert_frame_equal(expected_returns[factor["name"]], actual,
                                              check_exact=True, check_dtype=False)
            report_path = report(config.output)
            html = report_path.read_text()
            usage = pd.read_csv(config.output / "data_usage.csv")
            payload = json.loads((config.output / "report_data.json").read_text())
            self.assertEqual(payload["meta"]["industry_classification"], "aiq_zx35")
            self.assertTrue(payload["data_usage"])
            cap_usage = next(row for row in payload["data_usage"] if row["入口"] == "aiq_cap")
            self.assertIn("float_value_yuan", cap_usage["本次读取字段"].split(", "))
            self.assertIn("sector", usage.loc[usage["入口"].eq("aiq_industry"), "本次读取字段"].iloc[0])
            self.assertIn("float_value_yuan", usage.loc[usage["入口"].eq("aiq_cap"), "本次读取字段"].iloc[0])
            quality = pd.read_csv(config.output / "data_quality_summary.csv").set_index("指标")
            self.assertEqual(quality.loc["行业有效", "记录数"], 570)
            self.assertEqual(quality.loc["流通市值有效", "记录数"], 570)
            self.assertEqual(quality.loc["行业有效", "分母"], 600)
            self.assertTrue((config.output / "industry_distribution.csv").exists())


class AIQStockMasterIntegrationTests(unittest.TestCase):
    def test_union_retains_absent_securities_and_discloses_disagreeing_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"code": ["600001.SH", "600002.SH", "430001.BJ"],
                "ipo_date": [20100101, 20120101, 20150101], "delist_date": [20240201, None, None]}).to_csv(root / "aiq.csv", index=False)
            pd.DataFrame({"code": ["600001.SH"], "delisting_date": [20240102],
                          "last_trading_date_exchange": [20231229], "last_trading_date_stock": [20231228]}).to_csv(root / "delists.csv", index=False)
            old = pd.DataFrame({"code": ["600001.SH", "000001.SZ", "430001.BJ"],
                                "listing_date": pd.to_datetime(["2009-01-01", "2000-01-01", "2014-01-01"]),
                                "delisting_date": pd.NaT})
            old.to_parquet(root / "old.parquet", index=False)
            output = root / "combined.parquet"
            summary = prepare_stocks(root / "aiq.csv", root / "delists.csv", root / "old.parquet", output)
            actual = pd.read_parquet(output).set_index("code")
            self.assertEqual(set(actual.index), {"600001.SH", "600002.SH", "000001.SZ", "430001.BJ"})
            self.assertEqual(actual.loc["600001.SH", "listing_date"], pd.Timestamp("2010-01-01"))
            self.assertEqual(actual.loc["600001.SH", "delisting_date"], pd.Timestamp("2024-01-02"))
            self.assertEqual(actual.loc["430001.BJ", "listing_date"], pd.Timestamp("2014-01-01"))
            self.assertEqual(actual.loc["000001.SZ", "stock_source"], "public_exchange_master_fallback")
            self.assertEqual(actual.loc["430001.BJ", "stock_source"], "public_exchange_master_fallback")
            self.assertEqual(summary["fallback_rows"], 2)
            self.assertEqual(summary["listing_date_disagreements"], 1)
            self.assertEqual(summary["disagreement_rows"], 1)
            differences = pd.read_csv(output.with_suffix(".differences.csv"))
            self.assertEqual(differences.code.tolist(), ["600001.SH"])
            self.assertIn("delisting_date_existing", differences)


if __name__ == "__main__":
    unittest.main()
