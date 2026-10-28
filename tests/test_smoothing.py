"""Causal EMA behavior on a frozen rebalance grid, without return labels."""
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.smoothing import smooth_factor, smoothing_settings


class SmoothingTests(unittest.TestCase):
    @staticmethod
    def panel(values, *, codes=None, dates=None):
        dates = pd.date_range("2025-01-06", periods=len(values), freq="W-MON") if dates is None else pd.to_datetime(dates)
        frame = pd.DataFrame({"entry": dates, "code": codes or ["000001.SZ"] * len(values), "value": values})
        periods = frame[["entry"]].drop_duplicates().sort_values("entry").reset_index(drop=True)
        return frame, periods

    def test_none_is_exact_copy_and_does_not_mutate_input(self):
        frame, periods = self.panel([0., -2., np.nan, np.inf])
        frame.index = [3, 3, 5, 8]
        original = frame.copy(deep=True)
        result = smooth_factor(frame, periods)
        pd.testing.assert_frame_equal(result, original, check_exact=True)
        result.iloc[0, result.columns.get_loc("value")] = 100.
        pd.testing.assert_frame_equal(frame, original, check_exact=True)
        self.assertNotIn("smoothed_value", result)

    def test_hand_calculation_stock_isolation_and_input_row_order(self):
        first, periods = self.panel([10., 0., -10.])
        second, _ = self.panel([100., 200., 300.], codes=["000002.SZ"] * 3)
        frame = pd.concat([first, second], ignore_index=True).iloc[[4, 0, 3, 2, 5, 1]]
        frame.index = [9, 9, 2, 4, 1, 7]
        result = smooth_factor(frame, periods, {"method": "ema", "span": 4})
        np.testing.assert_allclose(result.smoothed_value, [140., 10., 100., -.4, 204., 6.])
        np.testing.assert_array_equal(result.smoothing_observations, [2, 1, 1, 3, 3, 2])
        np.testing.assert_array_equal(result.smoothing_reset, [False, True, True, False, False, False])
        pd.testing.assert_frame_equal(result[frame.columns], frame)

    def test_contiguous_ema_matches_recursive_pandas_definition(self):
        frame, periods = self.panel([3., 7., 2., -4., 0., 9.])
        result = smooth_factor(frame, periods, {"method": "ema", "span": 5, "min_periods": 2})
        expected = frame.value.ewm(span=5, adjust=False, min_periods=2).mean()
        np.testing.assert_allclose(result.smoothed_value, expected, equal_nan=True)

    def test_missing_and_nonfinite_update_stays_missing_and_resets_state(self):
        frame, periods = self.panel([10., np.nan, 20., 30., np.inf, -np.inf, 40., 50.])
        result = smooth_factor(frame, periods, {"method": "ema", "span": 4})
        np.testing.assert_allclose(result.smoothed_value, [10., np.nan, 20., 24., np.nan, np.nan, 40., 44.], equal_nan=True)
        np.testing.assert_array_equal(result.smoothing_observations, [1, 0, 1, 2, 0, 0, 1, 2])
        np.testing.assert_array_equal(result.smoothing_reset, [True, False, True, False, False, False, True, False])

    def test_absent_universe_period_resets_even_without_an_explicit_nan(self):
        frame, periods = self.panel([10., 999., 20., 30.])
        frame = frame.drop(index=1)
        result = smooth_factor(frame, periods, {"method": "ema", "span": 4, "min_periods": 2})
        np.testing.assert_allclose(result.smoothed_value, [np.nan, np.nan, 24.], equal_nan=True)
        np.testing.assert_array_equal(result.smoothing_observations, [1, 1, 2])
        np.testing.assert_array_equal(result.smoothing_reset, [True, True, False])

    def test_future_changes_do_not_change_prior_signals(self):
        frame, periods = self.panel([3., 5., 7., 9., 11.])
        settings = {"method": "ema", "span": 3, "min_periods": 2}
        original = smooth_factor(frame, periods, settings)
        changed = frame.copy()
        changed.loc[3:, "value"] = [-1e10, 1e12]
        actual = smooth_factor(changed, periods, settings)
        pd.testing.assert_frame_equal(actual.iloc[:3], original.iloc[:3], check_exact=True)
        prefix = smooth_factor(frame.iloc[:3], periods.iloc[:3], settings)
        pd.testing.assert_frame_equal(prefix, original.iloc[:3], check_exact=True)

    def test_monthly_and_holiday_grids_count_periods_not_elapsed_days(self):
        dates = ["2025-01-02", "2025-02-05", "2025-03-03", "2025-04-01"]
        frame, periods = self.panel([2., 4., 8., 16.], dates=dates)
        result = smooth_factor(frame, periods.iloc[::-1], {"method": "ema", "span": 3})
        np.testing.assert_allclose(result.smoothed_value, [2., 3., 5.5, 10.75])
        np.testing.assert_array_equal(result.smoothing_observations, [1, 2, 3, 4])

    def test_min_periods_is_consecutive_and_restarts_after_gap(self):
        frame, periods = self.panel([1., 2., np.nan, 4., 8., 16.])
        result = smooth_factor(frame, periods, {"method": "ema", "span": 3, "min_periods": 3})
        np.testing.assert_allclose(result.smoothed_value, [np.nan] * 5 + [11.], equal_nan=True)

    def test_span_one_preserves_finite_values_including_zeros_and_negatives(self):
        frame, periods = self.panel([0., -2., np.nan, 3., 1e308, -1e308])
        result = smooth_factor(frame, periods, {"method": "ema", "span": 1})
        np.testing.assert_allclose(result.smoothed_value, frame.value, equal_nan=True)
        frame, periods = self.panel([1e308, -1e308])
        result = smooth_factor(frame, periods, {"method": "ema", "span": 3})
        np.testing.assert_allclose(result.smoothed_value, [1e308, 0.])

    def test_smoothing_can_reduce_rank_changes_without_turnover_constraint(self):
        # One stable score competes with an oscillating noisy score. The EMA
        # dampens noise enough to reduce top-stock switches in this fixture.
        noisy, periods = self.panel([0., 2.] * 20)
        stable = noisy.assign(code="000002.SZ", value=.7)
        frame = pd.concat([noisy, stable], ignore_index=True)
        result = smooth_factor(frame, periods, {"method": "ema", "span": 4})
        def switches(column):
            chosen = result.loc[result.groupby("entry")[column].idxmax()].sort_values("entry").code
            return int((chosen.to_numpy()[1:] != chosen.to_numpy()[:-1]).sum())
        self.assertLess(switches("smoothed_value"), switches("value"))

    def test_invalid_settings_and_grid_keys_are_rejected(self):
        for settings in ({"method": "rolling"}, {"method": []}, {"span": 0}, {"span": True},
                         {"span": 1.5}, {"min_periods": 0}, {"span": 2, "min_periods": 3},
                         {"adjust": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                smoothing_settings(settings)
        frame, periods = self.panel([1., 2., 3.])
        settings = {"method": "ema"}
        for bad_frame, bad_periods in ((pd.concat([frame, frame.iloc[:1]]), periods),
                                      (frame, periods.iloc[1:]),
                                      (frame, pd.concat([periods, periods.iloc[:1]])),
                                      (frame.assign(code=None), periods),
                                      (frame.assign(entry=pd.NaT), periods),
                                      (frame.assign(smoothed_value=0), periods)):
            with self.subTest(rows=len(bad_frame)), self.assertRaises(ValueError):
                smooth_factor(bad_frame, bad_periods, settings)
        original = {"method": "ema", "span": 3}
        self.assertEqual(smoothing_settings(original), {"method": "ema", "span": 3, "min_periods": 1})
        self.assertEqual(original, {"method": "ema", "span": 3})

    def test_empty_panel_is_valid_and_retains_schema(self):
        frame, periods = self.panel([1.])
        actual = smooth_factor(frame.iloc[:0], periods, {"method": "ema"})
        self.assertTrue(actual.empty)
        self.assertEqual(actual.smoothed_value.dtype, np.dtype(float))
        self.assertEqual(actual.smoothing_observations.dtype, np.dtype(np.int64))
        self.assertEqual(actual.smoothing_reset.dtype, np.dtype(bool))


if __name__ == "__main__":
    unittest.main()
