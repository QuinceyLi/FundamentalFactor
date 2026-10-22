from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.audit import audit, source_columns
from fundamental_factor.config import RISK_FIELDS, load_config
from fundamental_factor.preprocess import control_matrix, prepare_signals, standardize
from fundamental_factor.panel import base_universe, weekly_calendar
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture, signal_fixture


class IndustryOnlyTests(unittest.TestCase):
    def test_industry_neutralization_is_within_industry_demeaning_without_cap(self):
        frame, _ = signal_fixture(weeks_count=2)
        frame = frame.drop(columns="float_mcap")
        signals, diagnostics = prepare_signals(frame, include_float_mcap=False)
        self.assertEqual(set(signals.variant), {"raw", "industry", "full"})
        self.assertNotIn("float_mcap", signals)
        self.assertFalse(diagnostics.status.eq("failed").any())
        for _, week in signals.groupby("entry"):
            raw = week[week.variant.eq("raw")].set_index("code")
            neutral = week[week.variant.eq("industry")].set_index("code")
            demeaned = raw.signal - raw.groupby("industry").signal.transform("mean")
            np.testing.assert_allclose(neutral.signal, standardize(demeaned.to_numpy()), atol=1e-12)
            np.testing.assert_allclose(neutral.groupby("industry").signal.mean(), 0., atol=1e-12)
        expected = set(map(tuple, frame[["entry", "code"]].to_numpy()))
        for _, rows in signals.groupby("variant"):
            self.assertEqual(set(map(tuple, rows[["entry", "code"]].to_numpy())), expected)

    def test_unused_cap_changes_cannot_change_any_signal_or_group(self):
        frame, _ = signal_fixture(weeks_count=1)
        first, _ = prepare_signals(frame, include_float_mcap=False)
        frame["float_mcap"] = np.resize([np.nan, np.inf, -100., 0.], len(frame))
        second, _ = prepare_signals(frame, include_float_mcap=False)
        selected = ["entry", "code", "variant", "signal", "percentile", "group"]
        pd.testing.assert_frame_equal(first[selected], second[selected])
        controls, names = control_matrix(frame.drop(columns="float_mcap"), full=True, include_float_mcap=False)
        self.assertNotIn("log_float_mcap", names)
        self.assertTrue(set(RISK_FIELDS).issubset(names))
        self.assertIn("SIZE", names)
        self.assertTrue(np.isfinite(controls).all())

    def test_full_pipeline_without_actual_market_cap_and_legacy_config_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            config = disk_fixture(Path(folder), weeks_count=4, stocks_count=80)
            legacy_digest = config.digest
            self.assertTrue(config.include_float_mcap)
            self.assertNotIn("neutralization", config.raw)
            self.assertEqual(load_config(config.path).digest, legacy_digest)
            self.assertIn("float_mcap", source_columns(config)["history"])
            raw = json.loads(config.path.read_text())
            raw["neutralization"] = {"include_float_mcap": False}
            config.path.write_text(json.dumps(raw), encoding="utf-8")
            history = pd.read_parquet(config.source("history"))
            history.drop(columns="float_mcap").to_parquet(config.source("history"), index=False)
            config = load_config(config.path)
            self.assertNotEqual(config.digest, legacy_digest)
            self.assertEqual(source_columns(config)["history"], ["date", "code", "industry", "is_st"])
            self.assertTrue(audit(config)["gate_passed"])
            result = run_research(config)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(len(result["factors"]), 2)
            features = pd.read_parquet(config.output / "features.parquet")
            controls = pd.read_parquet(config.output / "controls.parquet")
            self.assertNotIn("float_mcap", features)
            self.assertNotIn("float_mcap", controls)
            self.assertEqual(set(features.variant), {"raw", "industry", "full"})
            for factor in result["factors"]:
                self.assertEqual(set(factor["summary"]), {"raw", "industry", "full"})
                path = config.output / factor["output_dir"]
                performance = pd.read_csv(path / "performance.csv")
                self.assertEqual(set(performance.variant), {"raw", "industry", "full"})

    def test_legacy_mode_still_requires_cap_and_neutralization_config_is_strict(self):
        frame, _ = signal_fixture()
        with self.assertRaisesRegex(ValueError, "float_mcap"):
            prepare_signals(frame.drop(columns="float_mcap"))
        with tempfile.TemporaryDirectory() as folder:
            config = disk_fixture(Path(folder), weeks_count=2, stocks_count=40)
            raw = json.loads(config.path.read_text())
            for settings in (None, {}, {"include_float_mcap": "false"}, {"include_float_mcap": 0},
                             {"include_float_mcap": False, "extra": True}):
                with self.subTest(settings=settings):
                    raw["neutralization"] = settings
                    config.path.write_text(json.dumps(raw), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "neutralization"):
                        load_config(config.path)

    def test_exchange_selection_excludes_bj_and_does_not_require_its_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            config = disk_fixture(Path(folder), weeks_count=2, stocks_count=40)
            raw = json.loads(config.path.read_text())
            raw["universe"] = {"exchanges": ["SH", "SZ"]}
            config.path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_config(config.path)
            quotes = pd.read_feather(config.source("quotes"))
            bj = quotes.iloc[:1].copy()
            bj["code"] = "830001.BJ"
            pd.concat([quotes, bj], ignore_index=True).to_feather(config.source("quotes"))
            # The raw source contains BJ, but the selected universe does not
            # require its stock master, ST, industry or market-cap records.
            self.assertTrue(audit(config)["gate_passed"])
            stocks = pd.read_parquet(config.source("stocks"))
            bj_stock = stocks.iloc[:1].copy()
            bj_stock["code"] = "830001.BJ"
            pd.concat([stocks, bj_stock], ignore_index=True).to_parquet(config.source("stocks"), index=False)
            days = pd.read_csv(config.source("calendar"), parse_dates=["date"]).date
            weeks = weekly_calendar(days, pd.Timestamp("2015-01-05"), quotes.date.max())
            universe = base_universe(config, weeks)
            self.assertFalse(universe.code.str.endswith(".BJ").any())
            # No exchange key is injected into a legacy config.
            raw.pop("universe")
            config.path.write_text(json.dumps(raw), encoding="utf-8")
            legacy = load_config(config.path)
            self.assertIsNone(legacy.exchanges)
            self.assertNotIn("universe", legacy.raw)

    def test_exchange_configuration_is_explicit_and_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            config = disk_fixture(Path(folder), weeks_count=2, stocks_count=40)
            raw = json.loads(config.path.read_text())
            for settings in ({}, {"exchanges": []}, {"exchanges": "SH"},
                             {"exchanges": ["SH", "SH"]}, {"exchanges": ["HK"]},
                             {"exchanges": ["SH"], "unknown": True}):
                with self.subTest(settings=settings):
                    raw["universe"] = settings
                    config.path.write_text(json.dumps(raw), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "universe"):
                        load_config(config.path)


if __name__ == "__main__":
    unittest.main()
