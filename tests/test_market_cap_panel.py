"""Disk-to-backtest acceptance for quote-derived market-cap controls.

All observations are deterministic synthetic fixtures. No local market data or
network access is required, and raw factor results are compared independently
from the neutralized variant's control availability.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import load_config
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture


def proxy_fixture(root, *, frequency="weekly", weeks=4, stocks=60,
                  variants=None, controls=False):
    config = disk_fixture(root, weeks_count=weeks, stocks_count=stocks)
    quotes = pd.read_feather(config.source("quotes"))
    stock_number = quotes.code.str.split(".").str[0].astype(int).to_numpy()
    day_number = (quotes.date - quotes.date.min()).dt.days.to_numpy()
    quotes["turnover"] = .5 + .01 * stock_number
    quotes.to_feather(config.source("quotes"))
    raw_quotes = quotes[["date", "code"]].copy()
    raw_quotes["close"] = 10. + .2 * stock_number + .01 * day_number
    raw_quotes["volume"] = 50. + 2 * stock_number + 3 * day_number
    raw_path = Path(config.raw["data_root"]) / "market_cap_quotes.feather"
    raw_quotes.to_feather(raw_path)

    settings = json.loads(config.path.read_text())
    settings["data_policy"] = {"mode": "classic", "missing_returns": "available_case",
                               "empty_groups": "cash"}
    settings["preprocess"] = {"winsorize": False, "standardize": False}
    settings["neutralization"] = {"variants": variants or ["raw", "size"],
                                   "sample_policy": "independent"}
    settings["backtest"]["rebalance_frequency"] = frequency
    settings.pop("metrics", None)
    settings["sources"]["market_cap_quotes"] = str(raw_path)
    settings["neutralization"]["market_cap"] = {
        "method": "quote_proxy", "quotes_source": "market_cap_quotes",
        "turnover_source": "quotes", "close_column": "close",
        "volume_column": "volume", "turnover_column": "turnover",
        "volume_unit": "lot", "turnover_unit": "percent",
        "price_basis": "unadjusted",
    }
    if controls:
        history = pd.read_parquet(config.source("history")).drop(columns="float_mcap")
        history.to_parquet(config.source("history"), index=False)
    else:
        settings["sources"].update(history=None, risk=None)
    config.path.write_text(json.dumps(settings))
    return load_config(config.path)


def latest_expected_proxy(config, controls):
    """Independent oracle: exact-date join, then last record before each cutoff."""
    prices = pd.read_feather(config.source("market_cap_quotes"))
    rates = pd.read_feather(config.source("quotes"))[["date", "code", "turnover"]]
    observations = prices.merge(rates, on=["date", "code"], validate="one_to_one")
    expected = []
    for row in controls.itertuples(index=False):
        available = observations[(observations.code == row.code) &
                                 (observations.date <= row.cutoff)].sort_values("date")
        last = available.iloc[-1]
        expected.append(last.close * last.volume * 100 / (last.turnover / 100))
    return np.asarray(expected)


class MarketCapPanelTests(unittest.TestCase):
    def assert_size_residuals(self, features):
        for _, block in features.groupby(["factor", "entry"]):
            log_cap = np.log(block.market_cap_proxy.to_numpy())
            centered = log_cap - log_cap.mean()
            self.assertAlmostEqual(float(block.signal.sum()), 0., places=9)
            self.assertAlmostEqual(float(centered @ block.signal.to_numpy()), 0., places=9)

    def test_weekly_and_monthly_disk_pipeline_needs_no_industry_barra_or_actual_cap(self):
        for frequency, weeks in (("weekly", 4), ("monthly", 19)):
            with self.subTest(frequency=frequency), tempfile.TemporaryDirectory() as tmp:
                config = proxy_fixture(Path(tmp), frequency=frequency, weeks=weeks)
                result = run_research(config)
                features = pd.read_parquet(config.output / "features.parquet")
                controls = pd.read_parquet(config.output / "controls.parquet")
                periods = pd.read_parquet(config.output / "periods.parquet")
                checked = json.loads((config.output / "audit.json").read_text())
                self.assertTrue(checked["gate_passed"])
                self.assertEqual(result["sample"]["rebalance_frequency"], frequency)
                self.assertEqual(set(features.variant), {"raw", "size"})
                self.assertTrue(features.groupby(["factor", "entry", "variant"]).size().eq(60).all())
                for saved in (features, controls):
                    self.assertIn("market_cap_proxy", saved)
                    self.assertIn("market_cap_date", saved)
                    self.assertTrue(saved.market_cap_date.le(saved.cutoff).all())
                np.testing.assert_allclose(controls.market_cap_proxy,
                                           latest_expected_proxy(config, controls))
                raw = features[features.variant.eq("raw")]
                np.testing.assert_array_equal(raw.signal, raw.value)
                self.assert_size_residuals(features[features.variant.eq("size")])
                if frequency == "monthly":
                    self.assertEqual(periods.entry.dt.strftime("%Y-%m-%d").tolist(),
                                     ["2015-02-02", "2015-03-02", "2015-04-01"])
                for factor in result["factors"]:
                    performance = pd.read_csv(config.output / factor["output_dir"] / "portfolio_weekly.csv")
                    all_returns = performance[performance.portfolio.eq("ALL")]
                    self.assertEqual(len(all_returns), 2 * len(periods))
                    self.assertTrue(all_returns["return"].notna().all())

    def test_missing_cap_only_reduces_size_and_preserves_raw_backtest_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = proxy_fixture(Path(tmp), variants=["raw"])
            baseline = run_research(config)
            baseline_dir = config.output
            rates = pd.read_feather(config.source("quotes"))
            rates.loc[rates.code.isin([f"{i:06d}.SZ" for i in range(5)]), "turnover"] = 0.
            rates.to_feather(config.source("quotes"))
            prices = pd.read_feather(config.source("market_cap_quotes"))
            prices.loc[prices.code.isin([f"{i:06d}.SZ" for i in range(5, 8)]), "close"] = np.nan
            prices.to_feather(config.source("market_cap_quotes"))
            settings = json.loads(config.path.read_text())
            settings["neutralization"]["variants"] = ["raw", "size"]
            settings["output_dir"] = str(Path(tmp) / "missing_proxy_run")
            config.path.write_text(json.dumps(settings))
            config = load_config(config.path)
            result = run_research(config)
            coverage = pd.read_csv(config.output / "coverage.csv")
            self.assertTrue(coverage.variant_raw_count.eq(60).all())
            self.assertTrue(coverage.variant_size_count.eq(52).all())
            for original, changed in zip(baseline["factors"], result["factors"]):
                for name in ("ic_weekly.csv", "portfolio_weekly.csv", "performance.csv"):
                    expected = pd.read_csv(baseline_dir / original["output_dir"] / name)
                    actual = pd.read_csv(config.output / changed["output_dir"] / name)
                    pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                        actual[actual.variant.eq("raw")].reset_index(drop=True),
                        check_exact=True, check_dtype=False)

    def test_cutoff_uses_latest_same_day_inputs_without_future_or_cross_date_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = proxy_fixture(Path(tmp), weeks=3)
            prices = pd.read_feather(config.source("market_cap_quotes"))
            rates = pd.read_feather(config.source("quotes"))
            latest = prices.date.eq("2015-01-09") & prices.code.eq("000000.SZ")
            prices.loc[latest, ["close", "volume"]] = [40., 400.]
            rates.loc[rates.date.eq("2015-01-09") & rates.code.eq("000000.SZ"), "turnover"] = 2.
            # The rebalance-day value is dramatically different but unknowable
            # at the previous evening's signal cutoff.
            future = prices.date.ge("2015-01-12") & prices.code.eq("000000.SZ")
            prices.loc[future, ["close", "volume"]] = [1e7, 1e7]
            # This stock has a price/volume observation on Friday but turnover
            # only on older dates. Those dates must never be combined.
            prices = prices[~(prices.code.eq("000001.SZ") & prices.date.lt("2015-01-09"))]
            rates.loc[rates.code.eq("000001.SZ") & rates.date.eq("2015-01-09"), "turnover"] = np.nan
            # A new source observation on the rebalance day cannot rescue the
            # missing market-cap exposure at that day's signal cutoff.
            prices = prices[~(prices.code.eq("000002.SZ") & prices.date.lt("2015-01-12"))]
            prices.reset_index(drop=True).to_feather(config.source("market_cap_quotes"))
            rates.to_feather(config.source("quotes"))
            run_research(config)
            controls = pd.read_parquet(config.output / "controls.parquet")
            selected = controls[controls.entry.eq("2015-01-12")].set_index("code")
            self.assertAlmostEqual(selected.loc["000000.SZ", "market_cap_proxy"], 80_000_000.)
            self.assertEqual(selected.loc["000000.SZ", "market_cap_date"], pd.Timestamp("2015-01-09"))
            self.assertTrue(pd.isna(selected.loc["000001.SZ", "market_cap_proxy"]))
            self.assertTrue(pd.isna(selected.loc["000002.SZ", "market_cap_proxy"]))
            features = pd.read_parquet(config.output / "features.parquet")
            raw = features[features.entry.eq("2015-01-12") & features.variant.eq("raw")]
            self.assertTrue(raw.groupby("factor").size().eq(60).all())
            size = features[features.entry.eq("2015-01-12") & features.variant.eq("size")]
            self.assertFalse(size.code.isin(["000001.SZ", "000002.SZ"]).any())

    def test_industry_size_and_full_use_proxy_when_history_has_no_float_mcap(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = proxy_fixture(Path(tmp), weeks=3, controls=True,
                                   variants=["raw", "size", "industry_size", "full"])
            self.assertNotIn("float_mcap", pd.read_parquet(config.source("history")))
            run_research(config)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertEqual(set(features.variant), {"raw", "size", "industry_size", "full"})
            self.assertTrue(features.groupby(["factor", "entry", "variant"]).size().eq(60).all())
            for variant in ("size", "industry_size", "full"):
                block = features[features.variant.eq(variant)]
                self.assert_size_residuals(block)
                if variant != "size":
                    means = block.groupby(["factor", "entry", "industry"]).signal.mean()
                    np.testing.assert_allclose(means, 0., atol=1e-10)


if __name__ == "__main__":
    unittest.main()
