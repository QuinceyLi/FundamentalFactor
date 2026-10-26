"""Available-return aggregation must not change the frozen research sample."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import load_config
from fundamental_factor.panel import build_panel
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture


class AvailablePipelineTests(unittest.TestCase):
    def test_statistics_and_benchmark_use_available_labels_without_reforming_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=3)
            raw = json.loads(config.path.read_text())
            raw["data_policy"] = {"mode": "classic", "missing_returns": "report"}
            raw["neutralization"] = {"include_float_mcap": False, "variants": ["raw"]}
            config.path.write_text(json.dumps(raw))
            config = load_config(config.path)
            quotes = pd.read_feather(config.source("quotes"))
            mask = quotes.date.eq("2015-01-12") & quotes.code.eq("000003.SZ")
            quotes.loc[mask, "acc_open"] = np.nan
            quotes.to_feather(config.source("quotes"))
            build_panel(config)
            before_features = pd.read_parquet(config.output / "features.parquet")
            before_labels = pd.read_parquet(config.output / "labels.parquet")
            raw["output_dir"] = str(Path(tmp) / "available")
            raw["data_policy"].update(missing_returns="available_case", empty_groups="cash")
            config.path.write_text(json.dumps(raw))
            config = load_config(config.path)
            result = run_research(config)
            self.assertEqual(result["status"], "completed")
            pd.testing.assert_frame_equal(before_features, pd.read_parquet(config.output / "features.parquet"))
            labels = pd.read_parquet(config.output / "labels.parquet")
            pd.testing.assert_frame_equal(before_labels, labels)
            self.assertTrue(labels.loc[labels.unresolved, "return"].isna().all())
            benchmark = pd.read_csv(config.output / "benchmark.csv", parse_dates=["entry"])
            expected = labels.groupby("entry")["return"].mean()
            np.testing.assert_allclose(benchmark.set_index("entry")["return"], expected)
            self.assertFalse(benchmark.data_complete.all())
            self.assertTrue(benchmark.complete.all())
            manifest = json.loads((config.output / "panel_manifest.json").read_text())
            self.assertEqual(manifest["incomplete_benchmark_weeks"], 0)
            self.assertEqual(manifest["partial_benchmark_weeks"], 2)
            for card in result["factors"]:
                directory = config.output / card["output_dir"]
                weekly = pd.read_csv(directory / "portfolio_weekly.csv")
                self.assertTrue(weekly["return"].notna().all())
                self.assertTrue(weekly.nav.notna().all())
                self.assertTrue(weekly.query("portfolio != 'Q10-Q1'").excess_nav.notna().all())
                holdings = pd.read_parquet(directory / "holdings.parquet")
                positive = holdings.target_weight.gt(0)
                self.assertTrue(holdings.loc[positive & holdings["return"].isna(), "evaluation_weight"].eq(0).all())
                group = holdings[positive].copy()
                # Missing observations contribute no statistic, without turning
                # the original missing return itself into a zero observation.
                group["contribution"] = (group["return"] * group.evaluation_weight).fillna(0)
                measured = group.groupby(["entry", "variant", "portfolio"]).contribution.sum()
                w = weekly.query("portfolio != 'Q10-Q1' and n_stocks > 0").copy()
                w["entry"] = pd.to_datetime(w.entry)
                observed = w.set_index(["entry", "variant", "portfolio"])["return"]
                np.testing.assert_allclose(observed.sort_index(), measured.sort_index())


if __name__ == "__main__":
    unittest.main()
