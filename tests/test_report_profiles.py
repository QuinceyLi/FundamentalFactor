from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from fundamental_factor.config import RISK_FIELDS
from fundamental_factor.aiq_fill import filled_snapshot
from fundamental_factor.data import asof_snapshot
from fundamental_factor.report_profiles import (_cap_categories, _industry_names,
                                                 _last_risk_per_period, _profile,
                                                 build_profiles)


class ReportProfileTests(unittest.TestCase):
    def test_industry_mapping_does_not_relabel_legacy_class_numbers(self):
        frame = pd.DataFrame({"industry": ["AIQ_ZX_01", "class_1", None],
                              "industry_name": [None, None, None]})
        self.assertEqual(_industry_names(frame, {"AIQ_ZX_01": "银行"}).tolist(),
                         ["银行", "class_1", "未知行业"])

    def test_target_weights_missing_coverage_and_cap_median(self):
        frame = pd.DataFrame({"target_weight": [.25, .5], "evaluation_weight": [1., 0.],
                              "_industry": ["银行", "未知行业"], "float_mcap": [100., 300.],
                              "_cap_bin": [0, 4], **{field: [2., np.nan] for field in RISK_FIELDS}})
        p = _profile(frame)
        self.assertEqual(p['industries'], {"银行": .25, "未知行业": .5})
        self.assertEqual(p['cash'], .25)
        self.assertEqual(p['styles']['SIZE'], 2.)
        self.assertEqual(p['style_coverage']['SIZE'], .25)
        self.assertEqual(p['cap_median'], 300.)
        self.assertEqual(p['cap_bins'], [.25, 0., 0., 0., .5])
        self.assertEqual(p['industry_coverage'], .25)

    def test_cap_bins_keep_ties_together_and_exclude_invalid(self):
        frame = pd.DataFrame({'entry': [pd.Timestamp('2024-01-08')]*6,
                              'float_mcap': [10., 10., 30., 40., 0., np.nan]})
        bins = _cap_categories(frame)
        self.assertEqual(bins.iloc[0], bins.iloc[1])
        self.assertEqual(bins.iloc[3], 4)
        self.assertTrue(bins.iloc[4:].isna().all())

    def test_asof_keeps_explicit_null_and_does_not_take_next_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame=pd.DataFrame({'date':[20240102,20240105,20240108,20240116],
                                'code':['000001.SZ']*4,'SIZE':[1.,np.nan,999.,123.]})
            path=root/'risk.feather';frame.to_feather(path)
            cutoffs=pd.Series(pd.to_datetime(['2024-01-07 23:59:59','2024-01-14 23:59:59']))
            compact=_last_risk_per_period(path,cutoffs,['SIZE'])
            self.assertEqual(compact.date.dt.strftime('%Y%m%d').tolist(),['20240105','20240108'])
            self.assertTrue(pd.isna(compact.iloc[0]['SIZE']))

    def test_aiq_condensed_snapshots_match_filled_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'risk.feather'
            source=pd.DataFrame({'date':pd.to_datetime(['2024-01-02','2024-01-05','2024-01-07','2024-01-08','2024-01-12','2024-01-15']),
                                 'code':['000001.SZ']*6,'SIZE':[2.,np.nan,999.,0.,np.nan,777.],
                                 'BETA':[1.,-2.,999.,np.inf,np.nan,999.]})
            source.to_feather(path)
            base=pd.DataFrame({'code':['000001.SZ']*2,'cutoff':pd.to_datetime(['2024-01-07 23:59:59','2024-01-14 23:59:59'])})
            calendar=pd.Series(pd.bdate_range('2024-01-01','2024-01-15'))
            compact=_last_risk_per_period(path,base.cutoff,['SIZE','BETA'],aiq_fill=True,calendar=calendar)
            actual=asof_snapshot(base,compact,'report')
            expected=filled_snapshot(base,source,['SIZE','BETA'],calendar=calendar)
            for field in ['SIZE','BETA']:
                np.testing.assert_array_equal(actual[field],expected[field])
                np.testing.assert_array_equal(actual[field+'__donor'],expected[field+'_value_source_date'])

    def test_unsorted_source_batches_choose_actual_latest_field_donor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'risk.feather'
            source=pd.DataFrame({'date':pd.to_datetime(['2024-01-05','2024-01-06','2024-01-03','2024-01-08']),
                                 'code':['000001.SZ']*4,'SIZE':[2.,np.nan,1.,np.nan]})
            feather.write_feather(source,path,chunksize=2)
            compact=_last_risk_per_period(path,pd.Series(pd.to_datetime(['2024-01-10'])),['SIZE'],aiq_fill=True)
            self.assertEqual(compact.iloc[0]['SIZE'],2.)
            self.assertEqual(compact.iloc[0]['SIZE__donor'],pd.Timestamp('2024-01-05'))

    def test_complete_profile_uses_eligible_universe_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'factors'/'F';folder.mkdir(parents=True)
            entry=pd.Timestamp('2024-01-08');cutoff=pd.Timestamp('2024-01-07 23:59:59')
            universe=pd.DataFrame({'entry':[entry]*3,'cutoff':[cutoff]*3,'code':['000001.SZ','000002.SZ','000003.SZ'],
                                   'eligible':[True,True,False]})
            universe.to_parquet(root/'universe.parquet',index=False)
            controls=pd.DataFrame({'entry':[entry]*2,'code':['000001.SZ','000002.SZ'],
                                  'industry':['AIQ_ZX_01','AIQ_ZX_02'],'float_mcap':[100.,300.]})
            controls.to_parquet(root/'controls.parquet',index=False)
            pd.DataFrame({'industry_code':['AIQ_ZX_01','AIQ_ZX_02'],'industry_name':['银行','证券']}).to_csv(root/'industry_mapping.csv',index=False)
            risk=pd.DataFrame({'date':[20240105,20240105,20240108],
                               'code':['000001.SZ','000002.SZ','000001.SZ'],
                               **{field:[1.,3.,999.] for field in RISK_FIELDS}})
            risk.to_feather(root/'risk.feather')
            pd.DataFrame({'entry':[entry]*3,'code':['000001.SZ','000002.SZ','000002.SZ'],
                          'variant':['raw']*3,'portfolio':['Q10','Q1','Q10'],'target_weight':[1.,1.,0.],
                          'evaluation_weight':[0.,1.,1.],'is_exit':[False,False,True]}).to_parquet(folder/'holdings.parquet',index=False)
            p=build_profiles(root,{'sources':{'risk':str(root/'risk.feather')}},['F'])
            self.assertEqual(p['F']['raw']['2024-01-08']['Q10']['industries'],{'银行':1.})
            self.assertEqual(p['F']['raw']['2024-01-08']['Q10']['styles']['SIZE'],-1.)
            self.assertEqual(p['_universe']['2024-01-08']['styles']['SIZE'],0.)
            self.assertEqual(p['_universe']['2024-01-08']['industries'],{'银行':.5,'证券':.5})
            # The second generation must invalidate a cache after source edits.
            controls.loc[0,'industry']='AIQ_ZX_02'
            controls.to_parquet(root/'controls.parquet',index=False)
            changed=build_profiles(root,{'sources':{'risk':str(root/'risk.feather')}},['F'])
            self.assertEqual(changed['F']['raw']['2024-01-08']['Q10']['industries'],{'证券':1.})


if __name__ == '__main__':
    unittest.main()
