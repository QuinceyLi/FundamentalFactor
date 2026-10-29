from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_factor.audit import audit
from fundamental_factor.backtest import drift_turnover, evaluate_factor, performance_metrics, rank_ic
from fundamental_factor.cli import main
from fundamental_factor.config import DataGateError, RISK_FIELDS, load_config
from fundamental_factor.data import asof_snapshot, dates, fingerprint
from fundamental_factor.panel import build_panel, ensure_panel, make_labels, weekly_calendar
from fundamental_factor.preprocess import prepare_signals
from fundamental_factor.report import report
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture, signal_fixture


def changed_output(config, name="other"):
    raw = json.loads(config.path.read_text())
    raw["output_dir"] = str(config.path.parent / name)
    path = config.path.parent / f"{name}.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return load_config(path)


class TimingAndPreprocessingTests(unittest.TestCase):
    def test_dates_and_asof_preserve_explicit_missing(self):
        self.assertEqual(dates(pd.Series([20260102])).iloc[0], pd.Timestamp("2026-01-02"))
        right = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"]),
                              "code": ["000001.SZ"] * 3, "value": [1., np.nan, 99.]})
        left = pd.DataFrame({"cutoff": pd.to_datetime(["2026-01-01", "2026-01-03"]),
                             "code": ["000001.SZ"] * 2})
        result = asof_snapshot(left, right, "signal")
        self.assertEqual(result.value.iloc[0], 1.)
        self.assertTrue(pd.isna(result.value.iloc[1]))
        self.assertTrue((result.signal_date <= result.cutoff).all())

    def test_holiday_boundary_includes_initial_partial_week(self):
        calendar = pd.Series(pd.bdate_range("2025-01-01", "2025-01-24"))
        calendar = calendar[~calendar.isin(pd.to_datetime(["2025-01-06", "2025-01-07"]))]
        weeks = weekly_calendar(calendar, calendar.min(), calendar.max())
        self.assertEqual(weeks.entry.iloc[0], pd.Timestamp("2025-01-01"))
        self.assertEqual(weeks.entry.iloc[1], pd.Timestamp("2025-01-08"))
        self.assertTrue((weeks.cutoff < weeks.entry).all())
        self.assertTrue((weeks.exit > weeks.entry).all())
        self.assertNotIn("stage", weeks)

    def test_three_variants_have_identical_samples(self):
        frame, _ = signal_fixture()
        signals, _ = prepare_signals(frame)
        self.assertEqual(signals.variant.nunique(), 3)
        expected = set(map(tuple, frame[["entry", "code"]].to_numpy()))
        for _, rows in signals.groupby("variant"):
            self.assertEqual(set(map(tuple, rows[["entry", "code"]].to_numpy())), expected)

    def test_zero_negative_values_and_ties_are_preserved(self):
        frame, _ = signal_fixture(weeks_count=1)
        frame["value"] = np.r_[[-2.] * 10, [0.] * 70, [3.] * 20]
        signals, _ = prepare_signals(frame)
        # Raw signal is an affine transformation of the clipped input, so all zero
        # observations remain tied, even though their standardised score need not be zero.
        raw = signals[signals.variant.eq("raw")]
        zeros = raw[raw.code.isin(frame.loc[frame.value.eq(0), "code"])]
        self.assertEqual(zeros.signal.nunique(), 1)
        self.assertEqual(zeros.group.nunique(), 1)
        self.assertLess(raw.group.nunique(), 10)
        self.assertTrue((raw.loc[raw.code.isin(frame.loc[frame.value.lt(0), "code"]), "signal"] < zeros.signal.iloc[0]).all())

    def test_neutralisation_residuals_are_orthogonal_with_collinear_controls(self):
        frame, _ = signal_fixture(weeks_count=1)
        frame["SIZE"] = np.log(frame.float_mcap)
        frame["BETA"] = frame["SIZE"] * 2
        signals, diagnostics = prepare_signals(frame)
        merged = signals.merge(frame, on=["entry", "exit", "cutoff", "code"], suffixes=("", "_original"))
        neutral = merged[~merged.variant.eq("raw")]
        for variant, rows in neutral.groupby("variant"):
            y = rows.signal.to_numpy()
            industry = pd.get_dummies(rows.industry, dtype=float).to_numpy()
            controls = [np.ones(len(rows)), np.log(rows.float_mcap)]
            if "full" in variant:
                controls.extend(rows[field].to_numpy() for field in RISK_FIELDS)
            x = np.column_stack([*controls, industry])
            np.testing.assert_allclose(x.T @ y, 0., atol=1e-7)
        self.assertIsNotNone(diagnostics)

    def test_constant_signal_cannot_create_ic(self):
        frame, _ = signal_fixture(weeks_count=1)
        frame["value"] = 5.
        signals, _ = prepare_signals(frame)
        raw = signals[signals.variant.eq("raw")]
        self.assertLessEqual(raw.signal.nunique(), 1)
        self.assertTrue(np.isnan(rank_ic(raw.signal, np.arange(len(raw)))))


class BacktestArithmeticTests(unittest.TestCase):
    def test_rank_ic_handles_ties_and_distinguishes_pearson(self):
        self.assertAlmostEqual(rank_ic([-1, 0, 0, 2], [-2, 1, 1, 3]), 1.)
        x, y = np.arange(1, 6), np.arange(1, 6) ** 3
        self.assertAlmostEqual(rank_ic(x, y), 1.)
        self.assertLess(rank_ic(x, y, method="pearson"), .99)
        self.assertTrue(np.isnan(rank_ic([1, 1, 1], [1, 2, 3])))

    def test_drift_turnover_includes_exits_and_initial_construction(self):
        turnover, drift = drift_turnover({"A": .5, "B": .5})
        self.assertAlmostEqual(turnover, 1.)
        self.assertTrue(not drift or all(value == 0 for value in drift.values()))
        turnover, drift = drift_turnover({"A": .5, "B": .5}, {"A": .5, "B": .5}, {"A": .2, "B": 0.})
        self.assertAlmostEqual(drift["A"], 6 / 11)
        self.assertAlmostEqual(drift["B"], 5 / 11)
        self.assertAlmostEqual(turnover, 1 / 11)
        turnover, _ = drift_turnover({"C": 1.}, {"A": .5, "B": .5}, {"A": .2, "B": 0.})
        self.assertAlmostEqual(turnover, 2.)

    def test_performance_matches_hand_calculation(self):
        returns = pd.Series([.1, -.1])
        turnover = pd.Series([1., .1])
        excess = pd.Series([.08, -.12])
        result = performance_metrics(returns, turnover, excess)
        self.assertEqual(result["n_periods"], 2)
        self.assertAlmostEqual(result["total_return"], -.01)
        self.assertAlmostEqual(result["annual_return"], .99 ** 26 - 1)
        self.assertAlmostEqual(result["annual_volatility"], returns.std(ddof=1) * np.sqrt(52))
        self.assertAlmostEqual(result["information_ratio"], result["annual_return"] / result["annual_volatility"])
        self.assertAlmostEqual(result["max_drawdown"], .1)
        self.assertAlmostEqual(result["win_rate"], .5)
        self.assertAlmostEqual(result["annual_turnover"], 28.6)
        self.assertAlmostEqual(result["excess_total_return"], 1.08 * .88 - 1)

    @staticmethod
    def portfolio_fixture():
        entries = pd.date_range("2025-01-06", periods=3, freq="W-MON")
        weeks = pd.DataFrame({"entry": entries[:-1], "exit": entries[1:]})
        codes = [f"{i:06d}.SZ" for i in range(10)]
        records, labels = [], []
        for k, row in enumerate(weeks.itertuples(index=False)):
            returns = np.linspace(-.01, .08, 10) if k == 0 else np.linspace(.02, .04, 10)
            for j, code in enumerate(codes):
                records.append({"entry": row.entry, "exit": row.exit, "code": code,
                                "variant": "raw", "signal": float(j), "percentile": (j + 1) / 10,
                                "group": j + 1})
                labels.append({"entry": row.entry, "exit": row.exit, "code": code,
                               "return": returns[j], "unresolved": False, "trade_restricted": False})
        benchmark = weeks.assign(**{"return": [.02, -.01]})
        return pd.DataFrame(records), pd.DataFrame(labels), benchmark, weeks

    def test_equal_weight_excess_spread_and_nav_are_hand_checked(self):
        signals, labels, benchmark, weeks = self.portfolio_fixture()
        tables = evaluate_factor(signals, labels, benchmark, weeks)
        rows = tables["portfolio_weekly"]
        high = rows[rows.portfolio.eq("Q10")].sort_values("entry")
        low = rows[rows.portfolio.eq("Q1")].sort_values("entry")
        spread = rows[rows.portfolio.eq("Q10-Q1")].sort_values("entry")
        whole = rows[rows.portfolio.eq("ALL")].sort_values("entry")
        np.testing.assert_allclose(whole["return"], [.035, .03])
        self.assertAlmostEqual(high.nav.iloc[-1], 1.08 * 1.04)
        self.assertAlmostEqual(high.excess_nav.iloc[-1], 1.06 * 1.05)
        np.testing.assert_allclose(spread["return"], [.09, .02])
        self.assertAlmostEqual(spread.nav.iloc[-1], 1.09 * 1.02)
        np.testing.assert_allclose(spread.turnover, high.turnover.to_numpy() + low.turnover.to_numpy())
        self.assertAlmostEqual(spread.turnover.iloc[0], 2.)

    def test_missing_held_label_and_variant_sample_mismatch_are_rejected(self):
        signals, labels, benchmark, weeks = self.portfolio_fixture()
        with self.assertRaises(DataGateError):
            evaluate_factor(signals, labels.iloc[1:], benchmark, weeks)
        second = signals.copy()
        second["variant"] = "full"
        signals = pd.concat([signals, second.iloc[1:]], ignore_index=True)
        with self.assertRaises(DataGateError):
            evaluate_factor(signals, labels, benchmark, weeks)

    def test_gap_is_not_compressed_into_complete_history(self):
        result = performance_metrics(pd.Series([.1, np.nan, .2]), pd.Series([1., np.nan, 1.]))
        self.assertEqual(result["status"], "incomplete")
        for name in ["total_return", "annual_return", "max_drawdown", "annual_turnover"]:
            self.assertTrue(pd.isna(result[name]), name)


class PriceAndDataGateTests(unittest.TestCase):
    def test_total_return_levels_preserve_corporate_action_adjustment(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            entry, exit, code = pd.Timestamp("2015-01-05"), pd.Timestamp("2015-01-12"), "000003.SZ"
            quotes = pd.read_feather(config.source("quotes"))
            quotes.loc[quotes.date.eq(entry) & quotes.code.eq(code), "acc_open"] = 100.
            quotes.loc[quotes.date.eq(exit) & quotes.code.eq(code), "acc_open"] = 110.
            quotes.to_feather(config.source("quotes"))
            features = pd.DataFrame({"entry": [entry], "exit": [exit],
                                     "cutoff": [entry - pd.Timedelta(nanoseconds=1)], "code": [code],
                                     "delisting_date": [pd.NaT]})
            before = features.copy(deep=True)
            labels = make_labels(config, features)
            self.assertAlmostEqual(labels["return"].iloc[0], .1)
            pd.testing.assert_frame_equal(features, before)
            status = pd.read_parquet(config.source("status"))
            status = status[~(status.date.eq(exit) & status.code.eq(code))]
            status.to_parquet(config.source("status"), index=False)
            self.assertTrue(make_labels(config, features).unresolved.iloc[0])

    def test_suspended_exit_uses_verified_mark_and_terminal_zero_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            entry, exit, code = pd.Timestamp("2015-01-05"), pd.Timestamp("2015-01-12"), "000003.SZ"
            features = pd.DataFrame({"entry": [entry], "exit": [exit],
                                     "cutoff": [entry - pd.Timedelta(nanoseconds=1)], "code": [code],
                                     "delisting_date": [pd.NaT]})
            status = pd.read_parquet(config.source("status"))
            status.loc[status.date.eq(exit) & status.code.eq(code), "suspended"] = True
            status.to_parquet(config.source("status"), index=False)
            quotes = pd.read_feather(config.source("quotes"))
            quotes.loc[quotes.date.eq(exit - pd.Timedelta(days=3)) & quotes.code.eq(code), "acc_close"] = 105.
            quotes.to_feather(config.source("quotes"))
            labels = make_labels(config, features)
            self.assertFalse(labels.unresolved.iloc[0])
            self.assertAlmostEqual(labels["return"].iloc[0], .05)
            self.assertTrue(labels.trade_restricted.iloc[0])
            features["delisting_date"] = exit - pd.Timedelta(days=2)
            self.assertTrue(make_labels(config, features).unresolved.iloc[0])
            terminal_path = config.source("stocks").parent / "terminal.parquet"
            available = exit + pd.Timedelta(days=3)
            pd.DataFrame({"date": features.delisting_date, "available_date": [available],
                          "code": [code], "terminal_value": [0.]}).to_parquet(terminal_path, index=False)
            raw = json.loads(config.path.read_text())
            raw["sources"]["terminal"] = str(terminal_path)
            config.path.write_text(json.dumps(raw))
            terminal = make_labels(load_config(config.path), features)
            self.assertFalse(terminal.unresolved.iloc[0])
            self.assertEqual(terminal["return"].iloc[0], -1.)
            self.assertEqual(terminal.label_ready.iloc[0], available)

    def test_missing_metadata_stops_cli_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            raw = json.loads(config.path.read_text())
            raw["sources"]["history"] = None
            raw["price_contract"]["verified"] = False
            config.path.write_text(json.dumps(raw))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["build-panel", "--config", str(config.path)]), 2)
            self.assertFalse((config.output / "features.parquet").exists())
            html = (config.output / "report.html").read_text()
            self.assertTrue("缺" in html or "门槛" in html)

    def test_duplicates_and_missing_barra_fields_are_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            frame = pd.read_feather(config.source("fund"))
            pd.concat([frame, frame.iloc[:1]], ignore_index=True).to_feather(config.source("fund"))
            risk = pd.read_feather(config.source("risk"))
            risk.drop(columns=[RISK_FIELDS[-1]]).to_feather(config.source("risk"))
            checked = audit(config)
            self.assertFalse(checked["gate_passed"])
            self.assertTrue(any("duplicate keys" in text for text in checked["blocking"]))
            self.assertTrue(any("risk" in text for text in checked["blocking"]))

    def test_legacy_training_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            raw = json.loads(config.path.read_text())
            raw["study"] = {"training_weeks": 156}
            config.path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                load_config(config.path)


class IntegrationTests(unittest.TestCase):
    def test_five_week_pipeline_has_no_training_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            manifest = build_panel(config)
            self.assertEqual(manifest["status"], "ready")
            weeks = pd.read_parquet(config.output / "weeks.parquet")
            self.assertEqual(len(weeks), 5)
            self.assertNotIn("stage", weeks)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertTrue(features.industry.isin(["银行", "非银金融"]).any())
            self.assertTrue(features.entry.min() == weeks.entry.min())
            result = run_research(config)
            self.assertEqual(result["status"], "completed")
            self.assertEqual({factor["name"] for factor in result["factors"]}, {"ROIC_new", "fundnum_promotion"})

    def test_entry_filters_use_st_listing_age_and_suspension(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            entry = pd.Timestamp("2015-01-05")
            stocks = pd.read_parquet(config.source("stocks"))
            stocks.loc[stocks.code.eq("000005.SZ"), "listing_date"] = pd.Timestamp("2015-01-02")
            stocks.to_parquet(config.source("stocks"), index=False)
            history = pd.read_parquet(config.source("history"))
            history.loc[history.code.eq("000006.SZ"), "is_st"] = True
            history.to_parquet(config.source("history"), index=False)
            status = pd.read_parquet(config.source("status"))
            status.loc[status.date.eq(entry) & status.code.eq("000007.SZ"), "suspended"] = True
            status.to_parquet(config.source("status"), index=False)
            build_panel(config)
            labels = pd.read_parquet(config.output / "labels.parquet")
            codes = set(labels.loc[labels.entry.eq(entry), "code"])
            self.assertFalse({"000005.SZ", "000006.SZ", "000007.SZ"} & codes)
            self.assertIn("000000.SZ", codes)  # Banks are retained in the unified all-A universe.
            self.assertIn("000007.SZ", set(labels.loc[labels.entry.gt(entry), "code"]))

    def test_modified_panel_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            build_panel(config)
            with (config.output / "features.parquet").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(DataGateError):
                ensure_panel(config)

    def test_future_price_change_cannot_change_signals_or_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            build_panel(config)
            before = pd.read_parquet(config.output / "features.parquet")
            quotes = pd.read_feather(config.source("quotes"))
            quotes.loc[quotes.date.ge(pd.Timestamp("2015-01-12")), ["acc_open", "acc_close"]] *= 2
            quotes.to_feather(config.source("quotes"))
            other = changed_output(config)
            build_panel(other)
            after = pd.read_parquet(other.output / "features.parquet")
            pd.testing.assert_frame_equal(before, after)

    def test_benchmark_does_not_depend_on_factor_or_barra_missingness(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            build_panel(config)
            before = pd.read_csv(config.output / "benchmark.csv")
            roic = pd.read_feather(config.source("roic"))
            roic.loc[roic.code.eq("000010.SZ"), "ROIC_new"] = np.nan
            roic.to_feather(config.source("roic"))
            risk = pd.read_feather(config.source("risk"))
            risk.loc[risk.code.eq("000020.SZ"), RISK_FIELDS[0]] = np.nan
            risk.to_feather(config.source("risk"))
            other = changed_output(config)
            build_panel(other)
            pd.testing.assert_frame_equal(before, pd.read_csv(other.output / "benchmark.csv"))
            labels = pd.read_parquet(other.output / "labels.parquet")
            self.assertTrue(labels.code.eq("000010.SZ").any())
            self.assertTrue(labels.code.eq("000020.SZ").any())

    def test_unresolved_future_label_does_not_refilter_the_frozen_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=5)
            quotes = pd.read_feather(config.source("quotes"))
            mask = quotes.date.eq(pd.Timestamp("2015-01-12")) & quotes.code.eq("000003.SZ")
            quotes.loc[mask, "acc_open"] = np.nan
            quotes.to_feather(config.source("quotes"))
            with self.assertRaises(DataGateError):
                build_panel(config)
            features = pd.read_parquet(config.output / "features.parquet")
            self.assertTrue(features.loc[features.entry.eq(pd.Timestamp("2015-01-05")), "code"].eq("000003.SZ").any())
            labels = pd.read_parquet(config.output / "labels.parquet")
            self.assertTrue(labels.unresolved.any())
            self.assertFalse((config.output / "results.json").exists())

    def test_end_to_end_report_reproducibility_and_cache_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp))
            source_before = fingerprint(config.source("roic"))
            self.assertTrue(audit(config)["gate_passed"])
            first = run_research(config)
            repeated = run_research(config)
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(repeated, sort_keys=True))
            self.assertEqual(source_before, fingerprint(config.source("roic")))
            output = report(config.output)
            html = output.read_text()
            self.assertNotIn("<script src", html)
            payload = json.loads((config.output / "report_data.json").read_text())
            self.assertEqual(payload["meta"]["dataset_kind"], "synthetic")
            for factor in first["factors"]:
                factor_dir = config.output / factor["output_dir"]
                self.assertTrue((factor_dir / "report.html").exists())
                for name in ["ic_weekly", "ic_summary", "portfolio_weekly", "performance", "holdings", "annual_performance"]:
                    self.assertTrue((factor_dir / f"{name}.csv").exists(), name)
            # A source change cannot silently reuse cached signals.
            with config.source("roic").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(DataGateError):
                ensure_panel(config)


if __name__ == "__main__":
    unittest.main()
