"""Arithmetic, source alignment and point-in-time contracts for the quote proxy."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.data import asof_snapshot
from fundamental_factor.market_cap import market_cap_snapshot, quote_market_cap


class MarketCapTests(unittest.TestCase):
    def test_formula_units_invalid_inputs_and_overflow(self):
        values = pd.DataFrame([[10., 1000., 2.]] +
                              [[bad if i == j else [10., 1000., 2.][i] for i in range(3)]
                               for j in range(3) for bad in [0., -1., np.nan, np.inf, -np.inf]] +
                              [[1e308, 1000., 1.]], columns=['close', 'volume', 'turnover'])
        actual = quote_market_cap(values.close, values.volume, values.turnover)
        self.assertEqual(actual.market_cap_proxy.iloc[0], 50_000_000.)
        self.assertEqual(actual.market_cap_invalid_reason.iloc[0], '')
        self.assertTrue(actual.market_cap_proxy.iloc[1:].isna().all())
        self.assertTrue(actual.market_cap_invalid_reason.iloc[1:].ne('').all())
        self.assertEqual(actual.market_cap_invalid_reason.iloc[-1], 'invalid_result')

    def test_two_backward_snapshots_match_outer_join_then_asof(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = pd.DataFrame({'date': pd.to_datetime(['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07']),
                                'code': ['000001.SZ'] * 4, 'close': [10., 11., 99., 12.],
                                'volume': [1000., np.nan, 9000., 1000.]})
            turnover = pd.DataFrame({'date': pd.to_datetime(['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-08']),
                                     'code': ['000001.SZ'] * 4, 'turnover': [2., 2., 9., 2.]})
            raw.to_parquet(Path(tmp) / 'raw.parquet', index=False)
            turnover.to_parquet(Path(tmp) / 'turnover.parquet', index=False)
            config = self.config(tmp)
            base = pd.DataFrame({'code': ['000001.SZ'] * 7,
                                 'cutoff': pd.to_datetime(['2025-01-01 23:59:59', '2025-01-02 23:59:59',
                                                          '2025-01-03 23:59:59', '2025-01-05 23:59:59',
                                                          '2025-01-06 23:59:59', '2025-01-07 23:59:59',
                                                          '2025-01-08 23:59:59'])})
            actual = market_cap_snapshot(config, base)
            outer = raw.merge(turnover, on=['date', 'code'], how='outer', validate='one_to_one')
            calculated = quote_market_cap(outer.close, outer.volume, outer.turnover)
            outer['market_cap_proxy'] = calculated.market_cap_proxy
            expected = asof_snapshot(base, outer, 'market_cap')
            pd.testing.assert_series_equal(actual.market_cap_proxy, expected.market_cap_proxy)
            pd.testing.assert_series_equal(actual.market_cap_date, expected.market_cap_date)
            # Friday's null remains missing; Monday's valid full-day volume is not available on Sunday.
            self.assertTrue(actual.market_cap_proxy.iloc[2:4].isna().all())
            self.assertEqual(actual.market_cap_quote_date.iloc[3], pd.Timestamp('2025-01-03'))
            self.assertTrue(actual.market_cap_proxy.iloc[5:].isna().all())
            self.assertTrue(actual.market_cap_invalid_reason.iloc[5:].eq('source_date_mismatch').all())

    def test_same_source_and_absent_optional_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp)
            base = pd.DataFrame({'code': ['000001.SZ'], 'cutoff': [pd.Timestamp('2025-01-05 23:59:59')]})
            missing = market_cap_snapshot(config, base)
            self.assertTrue(missing.market_cap_proxy.isna().all())
            self.assertEqual(missing.market_cap_invalid_reason.iloc[0], 'missing_source_observation')
            raw = pd.DataFrame({'date': [pd.Timestamp('2025-01-03')], 'code': ['000001.SZ'],
                                'close': [10.], 'volume': [1000.], 'turnover': [2.]})
            raw.to_parquet(Path(tmp) / 'raw.parquet', index=False)
            config.market_cap['turnover_source'] = 'raw'
            actual = market_cap_snapshot(config, base)
            self.assertEqual(actual.market_cap_proxy.iloc[0], 50_000_000.)
            self.assertEqual(actual.market_cap_quote_date.iloc[0], actual.market_cap_turnover_date.iloc[0])
            raw.drop(columns=['volume']).to_parquet(Path(tmp) / 'raw.parquet', index=False)
            actual = market_cap_snapshot(config, base)
            self.assertTrue(actual.market_cap_proxy.isna().all())
            self.assertEqual(actual.market_cap_invalid_reason.iloc[0], 'invalid_volume')
            config.optional_control_sources = False
            with self.assertRaisesRegex(ValueError, 'missing market-cap fields'):
                market_cap_snapshot(config, base)

    @staticmethod
    def config(tmp):
        return SimpleNamespace(
            market_cap={'quotes_source': 'raw', 'turnover_source': 'turnover', 'close_column': 'close',
                        'volume_column': 'volume', 'turnover_column': 'turnover'},
            optional_control_sources=True, source=lambda name: Path(tmp) / f'{name}.parquet')


if __name__ == '__main__':
    unittest.main()
