"""Disk-to-result contracts for optional EMA, CITIC history and diagnostics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import DataGateError, load_config
from fundamental_factor.panel import ensure_panel
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture
from tests.test_configurable_framework import direct


def update(config, **settings):
    raw = json.loads(config.path.read_text())
    raw.update(settings)
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


class DiagnosticsPipelineTests(unittest.TestCase):
    def test_weekly_and_monthly_ema_preserve_inputs_and_export_exact_diagnostics(self):
        for frequency, periods in (("weekly", 5), ("monthly", 19)):
            with self.subTest(frequency=frequency), tempfile.TemporaryDirectory() as tmp:
                config = direct(disk_fixture(Path(tmp), weeks_count=periods, stocks_count=40), frequency=frequency)
                config = update(config, smoothing={"method": "ema", "span": 3, "min_periods": 1},
                                metrics={"turnover_warning_threshold": .2})
                result = run_research(config)
                features = pd.read_parquet(config.output / "features.parquet")
                grid = pd.read_parquet(config.output / "periods.parquet")
                np.testing.assert_array_equal(features.signal, features.smoothed_value)
                self.assertTrue(features.factor_date.le(features.cutoff).all())
                for factor in result["factors"]:
                    inputs = pd.read_parquet(config.output / "factor_inputs" / f"{factor['name']}.parquet")
                    self.assertEqual(len(inputs), len(grid) * 40)
                    ordered = inputs.sort_values(["code", "entry"])
                    expected = ordered.groupby("code", sort=False).value.transform(
                        lambda values: values.ewm(span=3, adjust=False).mean())
                    np.testing.assert_allclose(ordered.smoothed_value, expected)
                    block = features[features.factor.eq(factor["name"])].sort_values(["code", "entry"])
                    np.testing.assert_array_equal(block.value.to_numpy(), ordered.value.to_numpy())
                    np.testing.assert_allclose(block.signal.to_numpy(), expected.to_numpy())
                    self.assertEqual(factor["smoothing"], {"method": "ema", "span": 3, "min_periods": 1})
                    summary = factor["strategy_diagnostics"]["raw"]
                    self.assertEqual(summary["long_portfolio"], "Q10")
                    self.assertEqual(summary["short_portfolio"], "Q1")
                    self.assertEqual(summary["turnover_warning_threshold"], .2)
                    table = pd.read_csv(config.output / factor["output_dir"] / "portfolio_periodic.csv")
                    annualization = 52 if frequency == "weekly" else 12
                    for portfolio, value_column, name in (("Q10", "excess_return", "long_excess_sharpe"),
                                                          ("Q10-Q1", "return", "long_short_excess_sharpe")):
                        returns = table.loc[table.portfolio.eq(portfolio), value_column]
                        valid = returns[np.isfinite(returns)]
                        expected_sharpe = np.sqrt(annualization) * valid.mean() / valid.std(ddof=1)
                        self.assertAlmostEqual(summary[name], expected_sharpe, places=10)
                        self.assertEqual(summary[f"{name}_n_periods"], len(valid))
                    long = table[table.portfolio.eq("Q10")]
                    self.assertEqual(summary["long_turnover_observations"], len(grid) - 1)
                    ongoing = long[~long.initial_build.astype(bool)]
                    self.assertAlmostEqual(summary["long_mean_turnover_one_way"], ongoing.turnover.mean() / 2)
                    self.assertAlmostEqual(summary["long_high_turnover_fraction"],
                                           float((ongoing.turnover / 2 > .2).mean()))

    def test_latest_missing_factor_resets_ema_and_is_preserved_in_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = direct(disk_fixture(Path(tmp), weeks_count=5, stocks_count=40))
            source = pd.read_feather(config.source("roic"))
            missing = source.date.eq("2015-01-09") & source.code.eq("000000.SZ")
            self.assertEqual(int(missing.sum()), 1)
            source.loc[missing, "ROIC_new"] = np.nan
            source.to_feather(config.source("roic"))
            config = update(config, smoothing={"method": "ema", "span": 3, "min_periods": 2})
            run_research(config)
            inputs = pd.read_parquet(config.output / "factor_inputs" / "ROIC_new.parquet")
            code = inputs[inputs.code.eq("000000.SZ")].sort_values("entry").set_index("entry")
            self.assertTrue(pd.isna(code.loc["2015-01-12", "value"]))
            self.assertTrue(pd.isna(code.loc["2015-01-12", "smoothed_value"]))
            self.assertEqual(code.loc["2015-01-12", "factor_date"], pd.Timestamp("2015-01-09"))
            self.assertTrue(bool(code.loc["2015-01-19", "smoothing_reset"]))
            self.assertEqual(code.loc["2015-01-19", "smoothing_observations"], 1)
            self.assertTrue(pd.isna(code.loc["2015-01-19", "smoothed_value"]))
            expected = .5 * code.loc["2015-01-19", "value"] + .5 * code.loc["2015-01-26", "value"]
            self.assertAlmostEqual(code.loc["2015-01-26", "smoothed_value"], expected)
            features = pd.read_parquet(config.output / "features.parquet")
            selected = features[features.factor.eq("ROIC_new") & features.code.eq("000000.SZ")]
            self.assertFalse(selected.entry.isin(pd.to_datetime(["2015-01-12", "2015-01-19"])).any())
            coverage = pd.read_csv(config.output / "coverage.csv")
            first = coverage[coverage.entry.eq("2015-01-05")]
            self.assertTrue(first.factor_available.eq(40).all())
            self.assertTrue(first.signal_input_available.eq(0).all())
            self.assertTrue(first.smoothing_unavailable_count.eq(40).all())

    def test_ema_input_cache_is_fingerprinted_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = direct(disk_fixture(Path(tmp), weeks_count=3, stocks_count=40))
            config = update(config, smoothing={"method": "ema", "span": 3})
            run_research(config)
            manifest = ensure_panel(config)
            relative = "factor_inputs/ROIC_new.parquet"
            self.assertIn(relative, manifest["artifact_fingerprints"])
            path = config.output / relative
            table = pd.read_parquet(path)
            table.loc[0, "value"] += 1.
            table.to_parquet(path, index=False)
            with self.assertRaisesRegex(DataGateError, "冻结面板产物已改变.*factor_inputs"):
                ensure_panel(config)

    def test_explicit_none_and_omitted_smoothing_produce_identical_raw_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = direct(disk_fixture(Path(tmp), weeks_count=4, stocks_count=40))
            baseline = run_research(config)
            old_dir = config.output
            config = update(config, smoothing={"method": "none"}, output_dir=str(Path(tmp) / "explicit_none"))
            changed = run_research(config)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertNotIn("smoothed_value", features)
            self.assertFalse((config.output / "factor_inputs").exists())
            np.testing.assert_array_equal(features.signal, features.value)
            pd.testing.assert_frame_equal(features, pd.read_parquet(old_dir / "features.parquet"), check_exact=True)
            for before, after in zip(baseline["factors"], changed["factors"]):
                for filename in ("ic_periodic.csv", "portfolio_periodic.csv", "performance.csv"):
                    pd.testing.assert_frame_equal(pd.read_csv(old_dir / before["output_dir"] / filename),
                        pd.read_csv(config.output / after["output_dir"] / filename), check_exact=True)

    def test_diagnostic_configuration_validation_and_legacy_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = direct(disk_fixture(Path(tmp), weeks_count=2, stocks_count=40))
            before = copy.deepcopy(config.raw)
            digest = config.digest
            self.assertEqual(config.smoothing, {"method": "none", "span": 4, "min_periods": 1})
            self.assertEqual(config.industry["classification"], "legacy")
            self.assertEqual(config.raw, before)
            self.assertEqual(config.digest, digest)
            self.assertNotIn("smoothing", config.raw)
            self.assertNotIn("turnover_warning_threshold", config.raw["metrics"])
            raw = json.loads(config.path.read_text())
            for value in (0, -.1, True, "0.5", float("nan"), float("inf")):
                bad = copy.deepcopy(raw)
                bad["metrics"] = {"turnover_warning_threshold": value}
                config.path.write_text(json.dumps(bad))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "turnover_warning_threshold"):
                    load_config(config.path)
            valid = copy.deepcopy(raw)
            valid["metrics"] = {"turnover_warning_threshold": .75}
            config.path.write_text(json.dumps(valid))
            self.assertEqual(load_config(config.path).raw["metrics"]["turnover_warning_threshold"], .75)
            for smoothing in ({"method": "rolling"}, {"span": 0}, {"span": True},
                              {"span": 2, "min_periods": 3}, {"adjust": True}):
                bad = copy.deepcopy(raw)
                bad["smoothing"] = smoothing
                config.path.write_text(json.dumps(bad))
                with self.subTest(smoothing=smoothing), self.assertRaises(ValueError):
                    load_config(config.path)

    def test_separate_citic_history_preserves_unknowns_without_shrinking_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = direct(disk_fixture(Path(tmp), weeks_count=4, stocks_count=40))
            baseline = run_research(config)
            old_dir = config.output
            history = pd.read_parquet(config.source("history"))
            history["industry"] = "class_35"
            history.to_parquet(config.source("history"), index=False)
            industry = pd.DataFrame({"date": [pd.Timestamp("2015-01-02")] * 40,
                                      "code": [f"{i:06d}.SZ" for i in range(40)],
                                      "industry_raw": ["Banks"] + ["CI005001"] * 39})
            industry.loc[industry.code.eq("000001.SZ"), "date"] = pd.Timestamp("2015-01-09")
            invalid = pd.DataFrame({"date": pd.to_datetime(["2015-01-09", "2015-01-16"]),
                                    "code": ["000000.SZ"] * 2,
                                    "industry_raw": ["class_35", "NonbankFinan"]})
            industry = pd.concat([industry, invalid], ignore_index=True)
            path = Path(config.raw["data_root"]) / "citic_history.parquet"
            industry.to_parquet(path, index=False)
            raw = json.loads(config.path.read_text())
            raw["sources"]["industry"] = str(path)
            raw["industry"] = {"classification": "citic_l1", "source": "industry", "column": "industry_raw"}
            raw["neutralization"]["variants"] = ["raw", "industry"]
            raw["output_dir"] = str(Path(tmp) / "citic")
            config.path.write_text(json.dumps(raw))
            config = load_config(config.path)
            changed = run_research(config)
            controls = pd.read_parquet(config.output / "controls.parquet").set_index(["entry", "code"])
            first = controls.loc[(pd.Timestamp("2015-01-05"), "000000.SZ")]
            self.assertEqual(first.industry, "银行")
            self.assertEqual(first.industry_code, "CI005021")
            unknown = controls.loc[(pd.Timestamp("2015-01-12"), "000000.SZ")]
            self.assertTrue(pd.isna(unknown.industry))
            self.assertEqual(unknown.industry_mapping_status, "unmapped")
            self.assertEqual(unknown.industry_raw, "class_35")
            restored = controls.loc[(pd.Timestamp("2015-01-19"), "000000.SZ")]
            self.assertEqual(restored.industry, "非银行金融")
            early = controls.loc[(pd.Timestamp("2015-01-05"), "000001.SZ")]
            self.assertTrue(pd.isna(early.industry))
            self.assertTrue(pd.isna(early.industry_source_date))
            coverage = pd.read_csv(config.output / "coverage.csv")
            self.assertTrue(coverage.variant_raw_count.eq(40).all())
            for factor in changed["factors"]:
                selected = coverage[coverage.factor.eq(factor["name"])].sort_values("entry")
                self.assertEqual(selected.variant_industry_count.tolist(), [39, 39, 40, 40])
            for before, after in zip(baseline["factors"], changed["factors"]):
                for filename in ("ic_periodic.csv", "portfolio_periodic.csv", "performance.csv"):
                    actual = pd.read_csv(config.output / after["output_dir"] / filename)
                    pd.testing.assert_frame_equal(pd.read_csv(old_dir / before["output_dir"] / filename),
                        actual[actual.variant.eq("raw")].reset_index(drop=True), check_exact=True)


if __name__ == "__main__":
    unittest.main()
