"""End-to-end contracts for direct factors, independent samples and monthly runs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.config import RISK_FIELDS, load_config
from fundamental_factor.audit import audit
from fundamental_factor.research import run_research
from fundamental_factor.report import report
from tests.fixtures import disk_fixture, enable_execution_fixture


def direct(config, *, variants=None, frequency='weekly', output='direct'):
    raw = json.loads(config.path.read_text())
    raw['output_dir'] = str(config.path.parent / output)
    raw['data_policy'] = {'mode': 'classic', 'missing_returns': 'available_case', 'empty_groups': 'cash'}
    raw['preprocess'] = {'winsorize': False, 'standardize': False}
    raw['neutralization'] = {'include_float_mcap': False,
                             'variants': variants or ['raw'], 'sample_policy': 'independent'}
    raw['backtest']['rebalance_frequency'] = frequency
    raw.pop('metrics', None)
    config.path.write_text(json.dumps(raw))
    return load_config(config.path)


class ConfigurableFrameworkTests(unittest.TestCase):
    def test_future_only_factor_codes_do_not_block_an_older_holding_range(self):
        for arrival, expected_gate in [('2026-10-22', True), ('2015-01-02', False)]:
            with self.subTest(arrival=arrival), tempfile.TemporaryDirectory() as tmp:
                config = direct(disk_fixture(Path(tmp), weeks_count=2, stocks_count=40))
                source = pd.read_feather(config.source('roic'))
                extra = pd.DataFrame({'date': [pd.Timestamp(arrival)], 'code': ['999999.SZ'], 'ROIC_new': [1.]})
                pd.concat([source, extra], ignore_index=True).to_feather(config.source('roic'))
                checked = audit(config)
                self.assertEqual(checked['gate_passed'], expected_gate)
                self.assertEqual(checked['stock_attribute_check_end'], pd.Timestamp('2015-01-19'))
                if not expected_gate:
                    self.assertTrue(any('999999.SZ' in issue for issue in checked['blocking']))

    def test_frequency_defaults_and_configuration_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = disk_fixture(Path(tmp), weeks_count=2)
            self.assertEqual(config.rebalance_frequency, 'weekly')
            self.assertEqual(config.sample_policy, 'common')
            config = direct(config, frequency='monthly')
            self.assertEqual(config.raw['metrics'], {'annualization': 12, 'nw_lag': 3})
            valid = json.loads(config.path.read_text())
            changes = [({'backtest': {'rebalance_frequency': 'daily'}}, 'rebalance_frequency'),
                       ({'metrics': {'annualization': 52}}, 'annualization'),
                       ({'preprocess': {'winsorize': 'false'}}, 'boolean'),
                       ({'preprocess': {'standardize': 0}}, 'boolean'),
                       ({'neutralization': {'sample_policy': 'unknown'}}, 'sample_policy')]
            for changeset, message in changes:
                raw = json.loads(json.dumps(valid))
                for key, values in changeset.items():
                    raw.setdefault(key, {}).update(values)
                config.path.write_text(json.dumps(raw))
                with self.subTest(changes=changeset), self.assertRaisesRegex(ValueError, message):
                    load_config(config.path)

    def test_raw_outputs_do_not_change_when_neutral_variants_lose_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = direct(disk_fixture(Path(tmp), weeks_count=4, stocks_count=60))
            history = pd.read_parquet(config.source('history'))
            history.loc[history.code.isin([f'{i:06d}.SZ' for i in range(5)]), 'industry'] = None
            history.to_parquet(config.source('history'), index=False)
            risk = pd.read_feather(config.source('risk'))
            risk.loc[risk.code.isin([f'{i:06d}.SZ' for i in range(5, 10)]), RISK_FIELDS] = np.nan
            # A whole unavailable neutral period must not suppress raw signals.
            risk.loc[risk.date.eq(risk.date.min()), RISK_FIELDS] = np.nan
            risk.to_feather(config.source('risk'))
            baseline = run_research(config)
            base_dir = config.output
            config = direct(config, variants=['raw', 'industry', 'full'], output='independent')
            result = run_research(config)
            features = pd.read_parquet(config.output / 'features.parquet')
            raw = features[features.variant.eq('raw')]
            np.testing.assert_array_equal(raw.signal, raw.value)
            self.assertTrue(features.factor_date.le(features.cutoff).all())
            for a, b in zip(baseline['factors'], result['factors']):
                for filename in ('ic_weekly.csv', 'portfolio_weekly.csv', 'performance.csv'):
                    expected = pd.read_csv(base_dir / a['output_dir'] / filename)
                    actual = pd.read_csv(config.output / b['output_dir'] / filename)
                    pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                        actual[actual.variant.eq('raw')].reset_index(drop=True), check_exact=True, check_dtype=False)
            coverage = pd.read_csv(config.output / 'coverage.csv')
            self.assertTrue(coverage.variant_raw_count.eq(60).all())
            self.assertTrue(coverage.variant_industry_count.eq(55).all())
            self.assertTrue(coverage.variant_full_count.le(50).all())
            self.assertTrue(coverage.variant_full_count.eq(0).any())

    def test_missing_control_files_or_columns_keep_raw_and_disclose_failed_variants(self):
        for missing_kind in ('files', 'columns'):
            with self.subTest(kind=missing_kind), tempfile.TemporaryDirectory() as tmp:
                config = direct(disk_fixture(Path(tmp), weeks_count=2, stocks_count=40),
                                variants=['raw', 'industry', 'full'])
                if missing_kind == 'files':
                    raw = json.loads(config.path.read_text())
                    raw['sources'].update(history=None, risk=None)
                    config.path.write_text(json.dumps(raw))
                    config = load_config(config.path)
                else:
                    history = pd.read_parquet(config.source('history')).drop(columns=['industry'])
                    history.to_parquet(config.source('history'), index=False)
                    risk = pd.read_feather(config.source('risk')).drop(columns=RISK_FIELDS)
                    risk.to_feather(config.source('risk'))
                result = run_research(config)
                audit = json.loads((config.output / 'audit.json').read_text())
                self.assertTrue(audit['gate_passed'])
                self.assertTrue(any('原始版本' in warning for warning in audit['warnings']))
                for factor in result['factors']:
                    weekly = pd.read_csv(config.output / factor['output_dir'] / 'portfolio_weekly.csv')
                    self.assertTrue(weekly.loc[weekly.variant.eq('raw'), 'return'].notna().all())
                    unavailable = weekly[weekly.variant.ne('raw')]
                    self.assertTrue(unavailable['return'].isna().all())
                    self.assertTrue(unavailable.loc[unavailable.portfolio.ne('Q10-Q1'), 'reason'].eq('missing_signal').all())
                    self.assertFalse(unavailable.cash_weight.eq(1).any())
                report(config.output)

    def test_monthly_pipeline_and_execution_use_month_boundaries_and_annualization(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = direct(enable_execution_fixture(disk_fixture(Path(tmp), weeks_count=19, stocks_count=40)),
                            frequency='monthly')
            # Supply truly sparse monthly factor updates, not daily resampled averages.
            for name in ('roic', 'fund'):
                table = pd.read_feather(config.source(name))
                dates = table[['date']].drop_duplicates()
                last = dates.groupby(dates.date.dt.to_period('M')).date.max()
                table[table.date.isin(last)].to_feather(config.source(name))
            result = run_research(config)
            self.assertEqual(result['sample']['rebalance_frequency'], 'monthly')
            periods = pd.read_parquet(config.output / 'periods.parquet')
            pd.testing.assert_frame_equal(periods, pd.read_parquet(config.output / 'weeks.parquet'))
            self.assertEqual(list(periods.entry.dt.strftime('%Y-%m-%d')), ['2015-02-02', '2015-03-02', '2015-04-01'])
            self.assertEqual(list(periods.exit.dt.strftime('%Y-%m-%d')), ['2015-03-02', '2015-04-01', '2015-05-01'])
            features = pd.read_parquet(config.output / 'features.parquet')
            np.testing.assert_array_equal(features.signal, features.value)
            self.assertTrue(features.factor_date.le(features.cutoff).all())
            quotes = pd.read_feather(config.source('quotes')).set_index(['date', 'code'])
            labels = pd.read_parquet(config.output / 'labels.parquet')
            expected = [quotes.loc[(r.exit, r.code), 'acc_open'] / quotes.loc[(r.entry, r.code), 'acc_open'] - 1
                        for r in labels.itertuples(index=False)]
            np.testing.assert_allclose(labels['return'], expected)
            for factor in result['factors']:
                directory = config.output / factor['output_dir']
                periodic = pd.read_csv(directory / 'portfolio_periodic.csv')
                pd.testing.assert_frame_equal(periodic, pd.read_csv(directory / 'portfolio_weekly.csv'))
                all_returns = periodic[periodic.portfolio.eq('ALL')]['return'].to_numpy()
                performance = pd.read_csv(directory / 'performance.csv').set_index('portfolio')
                self.assertAlmostEqual(performance.loc['ALL', 'annual_return'],
                                       np.prod(1 + all_returns) ** (12 / len(all_returns)) - 1)
                self.assertTrue(factor['execution']['enabled'])
                executed = pd.read_csv(directory / 'execution_periodic.csv')
                self.assertEqual(set(executed.entry), set(periods.entry.dt.strftime('%Y-%m-%d')))
            report(config.output)
            payload = json.loads((config.output / 'report_data.json').read_text())
            self.assertEqual(payload['meta']['frequency'], 'monthly')
            self.assertEqual(payload['meta']['annualization'], 12)
            self.assertEqual(payload['config']['preprocess']['winsorize'], False)


if __name__ == '__main__':
    unittest.main()
