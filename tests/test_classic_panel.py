"""Classic analytics keep optional metadata uncertainty separate from price labels."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_factor.config import DataGateError, load_config
from fundamental_factor.panel import build_panel, ensure_panel, make_labels
from tests.fixtures import disk_fixture


def classic_config(root: Path, variants=None, missing_sources=()):
    config = disk_fixture(root, weeks_count=3)
    raw = json.loads(config.path.read_text())
    raw["data_policy"] = {"mode": "classic"}
    if variants is not None:
        raw["neutralization"] = {"include_float_mcap": False, "variants": variants}
    for source in missing_sources:
        raw["sources"][source] = None
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


class ClassicPanelTests(unittest.TestCase):
    def test_classic_cache_rejects_pre_classic_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp))
            manifest = build_panel(config)
            self.assertEqual(manifest["calculation_version"], 3)
            self.assertEqual(ensure_panel(config)["calculation_version"], 3)
            manifest.pop("calculation_version")
            (config.output / "panel_manifest.json").write_text(json.dumps(manifest, default=str))
            with self.assertRaisesRegex(DataGateError, "经典因子统计口径版本"):
                ensure_panel(config)

    def test_absent_st_and_status_do_not_erase_finite_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp), missing_sources=("status",))
            history = pd.read_parquet(config.source("history")).drop(columns="is_st")
            history.to_parquet(config.source("history"), index=False)
            manifest = build_panel(config)
            self.assertEqual(manifest["status"], "ready")
            self.assertEqual(manifest["unresolved_labels"], 0)
            self.assertEqual(manifest["unknown_st_base_rows"], 300)
            self.assertEqual(manifest["unknown_suspension_base_rows"], 300)
            labels = pd.read_parquet(config.output / "labels.parquet")
            self.assertTrue(labels["return"].notna().all())
            self.assertTrue(labels.entry_status_missing.all())
            self.assertTrue(labels.exit_status_missing.all())
            self.assertTrue(labels.trade_restriction_unknown.all())
            self.assertTrue(labels.entry_suspended.isna().all())
            self.assertFalse(labels.trade_restricted.any())

    def test_raw_only_requires_no_history_or_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp), variants=["raw"], missing_sources=("history", "risk", "status"))
            build_panel(config)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertEqual(set(features.variant), {"raw"})
            self.assertEqual(len(features), 600)
            self.assertTrue(features.history_date.isna().all())
            self.assertTrue(features.risk_date.isna().all())
            self.assertTrue(features.is_st_unknown.all())
            self.assertTrue(features.factor_date.le(features.cutoff).all())

    def test_industry_only_requires_no_risk_or_market_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp), variants=["raw", "industry"], missing_sources=("risk",))
            history = pd.read_parquet(config.source("history")).drop(columns=["float_mcap", "is_st"])
            history.to_parquet(config.source("history"), index=False)
            build_panel(config)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertEqual(set(features.variant), {"raw", "industry"})
            self.assertNotIn("float_mcap", features)
            self.assertEqual(len(features), 1200)
            for _, group in features.groupby(["factor", "entry"]):
                samples = group.groupby("variant").code.apply(set)
                self.assertEqual(samples["raw"], samples["industry"])

    def test_known_st_and_suspension_still_filter_and_unknown_st_is_not_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp))
            history = pd.read_parquet(config.source("history"))
            history.loc[history.code.eq("000003.SZ"), "is_st"] = True
            history.loc[history.code.eq("000005.SZ"), ["date", "is_st"]] = [pd.Timestamp("2015-01-06"), True]
            history.to_parquet(config.source("history"), index=False)
            status = pd.read_parquet(config.source("status"))
            status.loc[status.date.eq("2015-01-05") & status.code.eq("000004.SZ"), "suspended"] = True
            status.to_parquet(config.source("status"), index=False)
            build_panel(config)
            universe = pd.read_parquet(config.output / "universe.parquet")
            self.assertFalse(universe.loc[universe.code.eq("000003.SZ"), "eligible"].any())
            first = universe[universe.entry.eq("2015-01-05")].set_index("code")
            self.assertFalse(first.loc["000004.SZ", "eligible"])
            self.assertTrue(first.loc["000005.SZ", "eligible"])
            self.assertTrue(first.loc["000005.SZ", "is_st_unknown"])

    def test_missing_price_remains_missing_without_dropping_frozen_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp), missing_sources=("status",))
            prices = pd.read_feather(config.source("quotes"))
            prices.loc[prices.date.eq("2015-01-12") & prices.code.eq("000003.SZ"), "acc_open"] = np.nan
            prices.to_feather(config.source("quotes"))
            manifest = build_panel(config)
            self.assertEqual(manifest["status"], "ready")
            self.assertGreater(manifest["unresolved_labels"], 0)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertTrue((features.code.eq("000003.SZ") & features.entry.eq("2015-01-05")).any())
            labels = pd.read_parquet(config.output / "labels.parquet")
            first = labels[labels.code.eq("000003.SZ") & labels.entry.eq("2015-01-05")].iloc[0]
            self.assertTrue(first.unresolved)
            self.assertTrue(np.isnan(first["return"]))
            self.assertEqual(first.unresolved_reason, "invalid_exit_price")

    def test_delisting_with_observed_exit_price_is_disclosed_not_forced_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp), missing_sources=("status",))
            metadata = pd.DataFrame({"entry": [pd.Timestamp("2015-01-05")],
                                     "exit": [pd.Timestamp("2015-01-12")], "code": ["000003.SZ"],
                                     "delisting_date": [pd.Timestamp("2015-01-09")]})
            row = make_labels(config, metadata).iloc[0]
            self.assertFalse(row.unresolved)
            self.assertTrue(row.delisting_without_terminal)
            self.assertTrue(row.trade_restricted)
            self.assertTrue(np.isfinite(row["return"]))

    def test_zero_exit_quote_is_missing_but_explicit_zero_terminal_is_total_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp), missing_sources=("status",))
            prices = pd.read_feather(config.source("quotes"))
            prices.loc[prices.date.eq("2015-01-12") & prices.code.eq("000003.SZ"), "acc_open"] = 0.
            prices.to_feather(config.source("quotes"))
            metadata = pd.DataFrame({"entry": [pd.Timestamp("2015-01-05")],
                                     "exit": [pd.Timestamp("2015-01-12")], "code": ["000003.SZ"],
                                     "delisting_date": [pd.Timestamp("2015-01-09")]})
            missing = make_labels(config, metadata).iloc[0]
            self.assertTrue(missing.unresolved)
            self.assertTrue(np.isnan(missing["return"]))
            self.assertIn("invalid_exit_price", missing.unresolved_reason)
            terminal = Path(config.raw["data_root"]) / "terminal.parquet"
            pd.DataFrame({"date": [pd.Timestamp("2015-01-09")], "code": ["000003.SZ"],
                          "available_date": [pd.Timestamp("2015-01-09")], "terminal_value": [0.]}).to_parquet(terminal)
            raw = json.loads(config.path.read_text())
            raw["sources"]["terminal"] = str(terminal)
            config.path.write_text(json.dumps(raw))
            settled = make_labels(load_config(config.path), metadata).iloc[0]
            self.assertFalse(settled.unresolved)
            self.assertTrue(settled.terminal_applied)
            self.assertEqual(settled["return"], -1.)

    def test_suspension_carry_does_not_invent_delisting_exit_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = classic_config(Path(tmp))
            status = pd.read_parquet(config.source("status"))
            status.loc[status.date.eq("2015-01-12") & status.code.eq("000003.SZ"), "suspended"] = True
            status.to_parquet(config.source("status"), index=False)
            raw = json.loads(config.path.read_text())
            raw["price_contract"]["suspended_total_return"] = False
            config.path.write_text(json.dumps(raw))
            config = load_config(config.path)
            metadata = pd.DataFrame({"entry": [pd.Timestamp("2015-01-05")],
                                     "exit": [pd.Timestamp("2015-01-12")], "code": ["000003.SZ"],
                                     "delisting_date": [pd.NaT]})
            regular = make_labels(config, metadata).iloc[0]
            self.assertFalse(regular.unresolved)
            self.assertTrue(regular.exit_valuation_proxy)
            metadata["delisting_date"] = pd.Timestamp("2015-01-09")
            dead = make_labels(config, metadata).iloc[0]
            self.assertTrue(dead.unresolved)
            self.assertTrue(np.isnan(dead["return"]))
            self.assertEqual(dead.unresolved_reason, "delisting_without_terminal")


if __name__ == "__main__":
    unittest.main()
