import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from fundamental_factor.data import asof_snapshot
from fundamental_factor.industry import (INDUSTRY_FIELDS, LEGACY_FIELDS, industry_catalog,
                                         industry_from_exposures, normalize_industry,
                                         prepare_industry_history)


def exposures(fields, *, dates=None, codes=None, updates=None):
    result = pd.DataFrame(0., index=range(len(fields)), columns=INDUSTRY_FIELDS)
    for i, name in enumerate(fields):
        result.loc[i, name] = 1.
    result['date'] = pd.to_datetime(dates or ['2025-01-03'] * len(fields))
    result['code'] = codes or [f'{i:06d}.SZ' for i in range(len(fields))]
    result['updateTime'] = updates or ['2025-01-04 01:00:00'] * len(fields)
    return result


class CiticIndustryTests(unittest.TestCase):
    def test_catalog_has_thirty_unique_fields_and_chinese_labels(self):
        catalog = industry_catalog()
        self.assertEqual(len(catalog), 30)
        for field in ['industry_code', 'industry_name', 'industry_field', 'citic_level1_code']:
            self.assertEqual(catalog[field].nunique(), 30)
        self.assertEqual(catalog.set_index('industry_field').loc['Banks', 'industry_name'], '银行')
        self.assertEqual(catalog.set_index('industry_field').loc['DiverseFinan', 'industry_code'], 'CI005030')
        normalized = normalize_industry(catalog.industry_field)
        pd.testing.assert_series_equal(normalized.industry_name, catalog.industry_name.astype('string'))

    def test_unknown_native_classes_and_bare_numbers_are_never_guessed(self):
        values = pd.Series(['class_1', 'class_35', 35, '未定义', None, '', '   '])
        output = normalize_industry(values)
        self.assertTrue(output.industry_name.isna().all())
        self.assertEqual(output.industry_mapping_status.tolist(), ['unmapped'] * 4 + ['missing'] * 3)
        self.assertEqual(output.industry_raw.iloc[1], 'class_35')

    def test_aliases_codes_and_original_labels_are_preserved(self):
        output = normalize_industry(pd.Series([' Banks ', '银行', 'CI005021.WI', 'ci005021',
                                               '银行（中信）', 'CateringTourism', '电子元器件', '非银金融']))
        self.assertEqual(output.industry_name.tolist(), ['银行'] * 5 + ['消费者服务', '电子', '非银行金融'])
        self.assertEqual(output.industry_raw.iloc[0], ' Banks ')
        self.assertTrue(output.industry_mapping_status.eq('mapped').all())

    def test_onehot_requires_finite_binary_unique_membership(self):
        source = exposures(['Banks'] * 5)
        source.loc[1, 'Banks'] = 0.
        source.loc[2, 'Petroleum'] = 1.
        source.loc[3, 'Petroleum'] = np.nan
        source.loc[4, 'Banks'] = .5
        output = industry_from_exposures(source)
        self.assertEqual(output.industry.iloc[0], '银行')
        self.assertTrue(output.industry.iloc[1:].isna().all())
        self.assertTrue(output.industry_mapping_status.iloc[1:].eq('invalid_onehot').all())

    def test_legacy_field_fallback_and_conflicting_aliases(self):
        source = exposures(['ConsumerServices', 'Electronics', 'PowerEquipNewEnergy'])
        for new, old in LEGACY_FIELDS.items():
            source[old] = source[new]
            source[new] = np.nan
        output = industry_from_exposures(source)
        self.assertEqual(output.industry.tolist(), ['消费者服务', '电子', '电力设备及新能源'])
        source.loc[0, 'ConsumerServices'] = 0.
        output = industry_from_exposures(source)
        self.assertTrue(pd.isna(output.industry.iloc[0]))
        self.assertEqual(output.industry_mapping_status.iloc[0], 'conflicting_aliases')

    def test_valid_current_schema_takes_precedence_over_retired_zero_columns(self):
        source = exposures(['ConsumerServices', 'Electronics', 'PowerEquipNewEnergy'])
        for new, old in LEGACY_FIELDS.items():
            source[old] = 0.
        output = industry_from_exposures(source)
        self.assertTrue(output.industry_mapping_status.eq('mapped').all())
        self.assertEqual(output.industry.tolist(), ['消费者服务', '电子', '电力设备及新能源'])

    def test_schema_and_code_validation(self):
        source = exposures(['Banks'])
        with self.assertRaisesRegex(ValueError, 'Missing CITIC one-hot field'):
            industry_from_exposures(source.drop(columns='Banks'))
        source['code'] = ['000001.XSHE']
        self.assertEqual(industry_from_exposures(source).code.iloc[0], '000001.SZ')
        source['code'] = ['unknown']
        with self.assertRaisesRegex(ValueError, 'Noncanonical'):
            industry_from_exposures(source)

    def test_export_uses_conservative_update_day_and_keeps_newest_batch_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = exposures(['Banks', 'RealEstate', 'Banks', 'Banks'],
                dates=['2021-07-01', '2021-12-21', '2021-12-23', '2021-12-24'],
                codes=['000001.SZ'] * 4,
                updates=['2021-12-22 01:00:00', '2021-12-22 01:01:00', None, '2021-12-25 01:00:00'])
            path, output = root / 'source.feather', root / 'industry.parquet'
            source.to_feather(path)
            summary = prepare_industry_history(path, output)
            history = pd.read_parquet(output)
            self.assertEqual(summary['source_rows'], 4)
            self.assertEqual(summary['rows'], 2)
            self.assertEqual(summary['same_day_updates_consolidated'], 1)
            self.assertEqual(summary['quarantined_missing_update_time'], 1)
            self.assertEqual(history.date.tolist(), [pd.Timestamp('2021-12-23'), pd.Timestamp('2021-12-26')])
            self.assertEqual(history.industry.tolist(), ['房地产', '银行'])
            self.assertEqual(history.industry_observation_date.iloc[0], pd.Timestamp('2021-12-21'))
            targets = pd.DataFrame({'code': ['000001.SZ'] * 3,
                'cutoff': pd.to_datetime(['2021-07-04 23:59:59', '2021-12-21 23:59:59', '2021-12-24 23:59:59'])})
            matched = asof_snapshot(targets, history, 'citic')
            self.assertTrue(matched.industry.iloc[:2].isna().all())
            self.assertEqual(matched.industry.iloc[2], '房地产')
            self.assertTrue((root / 'industry_mapping.csv').is_file())
            self.assertTrue((root / 'industry.manifest.json').is_file())

    def test_verified_utc_timestamp_is_converted_before_available_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = exposures(['Banks'], dates=['2025-01-03'], updates=['2025-01-03 20:00:00'])
            source.to_feather(root / 'source.feather')
            prepare_industry_history(root / 'source.feather', root / 'industry.parquet', update_time_timezone='UTC')
            self.assertEqual(pd.read_parquet(root / 'industry.parquet').date.iloc[0], pd.Timestamp('2025-01-04'))

    def test_late_old_observation_cannot_roll_known_classification_backwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = exposures(['Banks', 'RealEstate'], codes=['000001.SZ'] * 2,
                dates=['2025-01-03', '2025-01-02'], updates=['2025-01-04 01:00:00', '2025-01-06 01:00:00'])
            source.to_feather(root / 'source.feather')
            summary = prepare_industry_history(root / 'source.feather', root / 'industry.parquet')
            output = pd.read_parquet(root / 'industry.parquet')
            self.assertEqual(len(output), 1)
            self.assertEqual(output.industry.iloc[0], '银行')
            self.assertEqual(summary['stale_late_observations_ignored'], 1)
            self.assertEqual(pd.read_csv(root / 'industry_quarantine.csv').reason.iloc[0], 'stale_late_observation')

    def test_conflicting_duplicate_original_snapshots_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = exposures(['Banks', 'RealEstate'], codes=['000001.SZ'] * 2)
            source.to_feather(root / 'source.feather')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                prepare_industry_history(root / 'source.feather', root / 'industry.parquet')
            self.assertFalse((root / 'industry.parquet').exists())

    def test_comparison_discloses_disagreement_without_overwriting_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            primary = exposures(['Banks', 'Banks'])
            comparison = exposures(['Banks', 'RealEstate'])
            primary.to_feather(root / 'primary.feather')
            comparison.to_feather(root / 'comparison.feather')
            summary = prepare_industry_history(root / 'primary.feather', root / 'industry.parquet', root / 'comparison.feather')
            self.assertEqual(summary['comparison']['matched_rows'], 2)
            self.assertEqual(summary['comparison']['disagreements'], 1)
            self.assertTrue(pd.read_parquet(root / 'industry.parquet').industry.eq('银行').all())


if __name__ == '__main__':
    unittest.main()
