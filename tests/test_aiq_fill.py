"""AIQ fill parity, causal provenance, events and explicit daily grids."""
import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.aiq_fill import (FILL_RULES, apply_fill, field_policy,
                                        fill_long_records, filled_snapshot)


class AIQFillTests(unittest.TestCase):
    def setUp(self):
        self.days = pd.date_range("2025-01-01", periods=6)

    def test_rule_names_exact(self):
        self.assertEqual(field_policy("volume"), "zero")
        self.assertEqual(field_policy("turnover"), "keep")
        self.assertEqual(field_policy("CAGR"), "keep")
        self.assertEqual(field_policy("vwap_210toe"), "last")
        self.assertEqual(field_policy("twap_30toe"), "last")
        for field in ("last", "close", "acc_open", "acc_close", "circulated_market_value", "factor", "cagr"):
            self.assertEqual(field_policy(field), "ffill")
        self.assertEqual(field_policy("close", {"close": "keep"}), "keep")
        with self.assertRaises(ValueError):
            field_policy("x", {"x": "bfill"})

    def test_isolated_original_apply_fill_random_parity(self):
        """Compile only the inspected pure function, never import AIQ modules."""
        path = Path("/data/AIQ/Tools/datahub.py")
        if not path.is_file():
            self.skipTest("Local AIQ source not available; deterministic contract tests still run")
        tree = ast.parse(path.read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_fill")
        # The extracted body uses only pandas operations and policy constants.
        self.assertTrue(all(not isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(fn)))
        fn.returns = None
        for arg in fn.args.args:
            arg.annotation = None
        isolated = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        scope = {"FILL_ZERO": "zero", "FILL_KEEP": "keep", "FILL_LAST": "last"}
        exec(compile(isolated, "<isolated AIQ pure _apply_fill>", "exec"), scope)
        rng = np.random.default_rng(20261028)
        for repeat in range(12):
            values = pd.DataFrame(rng.normal(size=(20, 7)))
            values[rng.random(values.shape) < .4] = np.nan
            values.iloc[:2] = np.nan
            values.iloc[4, :3] = [0., np.inf, -np.inf]
            reference = pd.DataFrame(rng.normal(size=(17, 5))).ffill()
            for policy in ("zero", "keep", "ffill", "last", "unknown_falls_back_like_source"):
                for last in (None, reference):
                    expected = scope["_apply_fill"](values.copy(), policy, last)
                    actual = apply_fill(values.copy(), policy, last)
                    pd.testing.assert_frame_equal(actual, expected)

    def test_random_long_grid_matches_matrix_and_records_origins(self):
        rng = np.random.default_rng(17)
        fields = ["volume", "turnover", "close", "vwap_30toe", "last"]
        keys = pd.MultiIndex.from_product([self.days, ["A", "B", "C"]], names=["date", "code"])
        frame = keys.to_frame(index=False)
        for field in fields:
            frame[field] = rng.normal(size=len(frame))
            frame.loc[rng.random(len(frame)) < .35, field] = np.nan
        frame = frame.drop(index=[0, 1, 2, 4, 8]).sample(frac=1., random_state=4)
        original = frame.copy(deep=True)
        actual = fill_long_records(frame, fields, calendar=self.days)
        raw_wide = {field: frame.pivot(index="date", columns="code", values=field).reindex(self.days)
                    for field in fields}
        last = raw_wide["last"].ffill()
        for field in fields:
            expected = apply_fill(raw_wide[field], field_policy(field), last)
            observed = actual.pivot(index="date", columns="code", values=field)
            pd.testing.assert_frame_equal(observed, expected, check_names=False, check_freq=False)
            self.assertFalse((actual[f"{field}_value_source_date"] > actual.date).any())
        pd.testing.assert_frame_equal(frame, original)

    def test_first_observation_zero_explicit_na_and_infinity(self):
        frame = pd.DataFrame({"date": [self.days[1], self.days[2], self.days[4]],
            "code": ["A"] * 3, "volume": [np.nan, -2., np.inf], "close": [np.nan, 7., np.nan]})
        actual = fill_long_records(frame, ["volume", "close"], calendar=self.days)
        np.testing.assert_equal(actual.volume.to_numpy(), [np.nan, np.nan, -2., 0., np.inf, 0.])
        np.testing.assert_equal(actual.close.to_numpy(), [np.nan, np.nan, 7., 7., 7., 7.])
        self.assertTrue(actual.loc[[3, 5], "volume_filled"].all())
        self.assertTrue(actual.loc[[3, 5], "volume_value_source_date"].isna().all())
        self.assertEqual(actual.volume_fill_anchor_date.iloc[5], self.days[4])
        self.assertEqual(actual.close_value_source_date.iloc[4], self.days[2])
        self.assertTrue(actual.close_original_missing.iloc[4])

    def test_keep_daily_grid_differs_from_latest_event(self):
        source = pd.DataFrame({"date": [self.days[0]], "code": ["A"], "turnover": [2.]})
        base = pd.DataFrame({"cutoff": [self.days[2]], "code": ["A"]})
        event = filled_snapshot(base, source, ["turnover"], prefix="factor")
        grid = filled_snapshot(base, source, ["turnover"], prefix="factor", calendar=self.days)
        self.assertEqual(event.turnover.iloc[0], 2.)
        self.assertTrue(pd.isna(grid.turnover.iloc[0]))
        self.assertEqual(grid.factor_record_date.iloc[0], self.days[0])
        self.assertEqual(grid.factor_grid_date.iloc[0], self.days[2])
        self.assertTrue(grid.turnover_original_missing.iloc[0])

    def test_explicit_tombstone_ffills_value_but_not_status(self):
        source = pd.DataFrame({"date": [self.days[0], self.days[2]], "code": ["A", "A"],
            "value": [10., np.nan], "status": ["valid", "explicit_missing"], "reason": ["", "missing_update"]})
        base = pd.DataFrame({"cutoff": [self.days[3], self.days[1]], "code": ["A", "A"]}, index=[42, 7])
        actual = filled_snapshot(base, source, ["value"], prefix="factor")
        self.assertEqual(actual.index.tolist(), [42, 7])
        self.assertEqual(actual.value.tolist(), [10., 10.])
        self.assertEqual(actual.status.tolist(), ["explicit_missing", "valid"])
        self.assertEqual(actual.reason.tolist(), ["missing_update", ""])
        self.assertEqual(actual.factor_date.iloc[0], self.days[2])
        self.assertEqual(actual.value_value_source_date.iloc[0], self.days[0])
        self.assertTrue(actual.value_filled.iloc[0])
        self.assertFalse(actual.value_filled.iloc[1])

    def test_last_reference_including_earlier_first_target_and_fallback(self):
        source = pd.DataFrame({"date": [self.days[2], self.days[4]], "code": ["A", "A"],
                               "vwap_30toe": [20., np.nan]})
        reference = pd.DataFrame({"date": [self.days[0], self.days[3]], "code": ["A", "A"], "last": [10., 30.]})
        actual = fill_long_records(source, ["vwap_30toe"], calendar=self.days, fill_source=reference)
        self.assertEqual(actual.vwap_30toe.tolist(), [10., 10., 20., 30., 30., 30.])
        self.assertEqual(actual.vwap_30toe_value_source_date.iloc[4], self.days[3])
        fallback = fill_long_records(source, ["vwap_30toe"], calendar=self.days)
        self.assertTrue(fallback.vwap_30toe.iloc[:2].isna().all())
        self.assertEqual(fallback.vwap_30toe.iloc[2:].tolist(), [20.] * 4)
        self.assertEqual(fallback.vwap_30toe_fill_method.iloc[4], "ffill")

    def test_nullable_strings_booleans_and_no_future_backfill(self):
        source = pd.DataFrame({"date": [self.days[1], self.days[3]], "code": ["A", "A"],
            "industry": pd.Series(["银行", pd.NA], dtype="string"),
            "is_st": pd.Series([False, pd.NA], dtype="boolean")})
        actual = fill_long_records(source, ["industry", "is_st"], calendar=self.days)
        self.assertTrue(actual.loc[0, ["industry", "is_st"]].isna().all())
        self.assertEqual(actual.industry.iloc[1:].tolist(), ["银行"] * 5)
        self.assertEqual(actual.is_st.iloc[1:].tolist(), [False] * 5)
        self.assertEqual(str(actual.industry.dtype), "string")
        self.assertEqual(str(actual.is_st.dtype), "boolean")
        self.assertTrue(actual.is_st_original_missing.iloc[3])

    def test_calendar_drops_off_grid_only_when_explicit(self):
        source = pd.DataFrame({"date": [self.days[0], self.days[1]], "code": ["A", "A"], "value": [1., 2.]})
        base = pd.DataFrame({"cutoff": [self.days[2]], "code": ["A"]})
        event = filled_snapshot(base, source, ["value"])
        grid = filled_snapshot(base, source, ["value"], calendar=[self.days[0], self.days[2]])
        self.assertEqual(event.value.iloc[0], 2.)
        self.assertEqual(grid.value.iloc[0], 1.)
        early = filled_snapshot(base.assign(cutoff=self.days[0]-pd.Timedelta(1,"ns")), source, ["value"], calendar=self.days)
        self.assertTrue(early.value.isna().all())
        self.assertTrue(early.source_grid_date.isna().all())

    def test_empty_sources_extra_codes_and_duplicate_keys(self):
        source = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "code": pd.Series(dtype="string"),
                               "close": pd.Series(dtype=float)})
        base = pd.DataFrame({"cutoff": [self.days[1]], "code": ["A"]})
        actual = filled_snapshot(base, source, ["close"], calendar=self.days)
        self.assertTrue(actual.close.isna().all())
        grid = fill_long_records(source, ["close"], calendar=self.days, codes=["A"])
        self.assertEqual(len(grid), len(self.days))
        self.assertTrue(grid.close.isna().all())
        dup = pd.DataFrame({"date": [self.days[0]]*2, "code": ["A"]*2, "close": [1.,2.]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            filled_snapshot(base, dup, ["close"])

    def test_snapshot_calendar_cutoff_between_days_never_uses_later_observation(self):
        source = pd.DataFrame({"date": [20250101,20250103], "code": ["A","A"], "close": [1.,3.]})
        base = pd.DataFrame({"cutoff": [pd.Timestamp('2025-01-03')-pd.Timedelta(1,'ns')], "code": ["A"]})
        actual = filled_snapshot(base, source, ["close"], calendar=[20250101,20250103])
        self.assertEqual(actual.close.iloc[0], 1.)
        self.assertEqual(actual.source_grid_date.iloc[0], self.days[0])
        self.assertEqual(actual.close_value_source_date.iloc[0], self.days[0])


if __name__ == "__main__":
    unittest.main()
