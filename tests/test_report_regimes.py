from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from fundamental_factor.report import build_regimes

class ReportRegimeTests(unittest.TestCase):
    def test_label_excludes_entry_day_and_future_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.csv'
            dates=pd.bdate_range('2023-01-02',periods=160)
            values=np.linspace(100,150,160)
            frame=pd.DataFrame({'date':dates,'close':values});frame.to_csv(path,index=False)
            entry=str(dates[140].date())
            first,_=build_regimes([entry],path)
            frame.loc[140:,'close']=1e6;frame.to_csv(path,index=False)
            second,_=build_regimes([entry],path)
            self.assertEqual(first,second)
            self.assertEqual(first[entry]['date'],str(dates[139].date()))
            self.assertEqual(first[entry]['state'],'bull')

    def test_warmup_absent_source_and_stale_data_stay_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.csv'
            entries=['2023-01-03','2024-01-01']
            absent,meta=build_regimes(entries,path)
            self.assertEqual(meta['status'],'missing_source')
            self.assertTrue(all(r['state']=='unknown' for r in absent.values()))
            pd.DataFrame({'date':pd.bdate_range('2023-01-02',periods=121),'close':np.arange(121)+100}).to_csv(path,index=False)
            got,_=build_regimes(entries,path)
            self.assertTrue(all(r['state']=='unknown' for r in got.values()))

    def test_long_holiday_inside_observed_history_uses_preholiday_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'index.csv'
            days=pd.bdate_range('2023-01-02',periods=160)
            frame=pd.DataFrame({'date':days,'close':np.arange(160)+100})
            frame=frame.drop(index=range(130,140));frame.to_csv(path,index=False)
            entry=str(days[140].date());got,_=build_regimes([entry],path)
            self.assertEqual(got[entry]['date'],str(days[129].date()))
            self.assertEqual(got[entry]['state'],'bull')
