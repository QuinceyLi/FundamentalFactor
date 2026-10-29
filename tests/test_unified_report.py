"""Migration of classic/available/optional/cap/diagnostic report guarantees.

Tests assert frozen inputs and numerical outputs instead of the deleted renderer's
private HTML snippets. Browser layout and controls are checked separately.
"""
from pathlib import Path
import base64
import gzip
import json
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fundamental_factor.backtest import evaluate_factor
from fundamental_factor.report import report
from tests.report_fixture import settings, create_run, render, write_json
from tests.test_backtest_edges import _fixture
from tests.test_turnover_sharpe import fixture as turnover_fixture


class UnifiedReportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = create_run(self.root, audit={"warnings": ["ST 未知"], "execution_gate_passed": False,
            "execution_blocking": ["缺少公司行动"]}, quality={"quality_status": "unresolved_labels",
            "base_rows": 30, "unknown_st_base_rows": 12, "unresolved_labels": 1, "label_coverage": 29/30},
            execution={"requested": True, "enabled": False, "status": "skipped", "blocking": ["缺少公司行动"]})

    def test_optional_data_gaps_do_not_hide_current_theory(self):
        data, _ = render(self.root)
        self.assertFalse(data["issues"])
        self.assertTrue(data["factors"][0]["portfolio"])
        self.assertEqual(data["quality"]["unresolved_labels"], 1)
        self.assertEqual(data["quality"]["unknown_st_base_rows"], 12)
        self.assertEqual(data["audit"]["warnings"], ["ST 未知"])

    def test_single_report_replaces_factor_pages_with_navigation(self):
        data, html = render(self.root)
        redirect = (self.folder / "report.html").read_text()
        self.assertIn("../../report.html", redirect)
        self.assertNotIn('id="report-data"', redirect)
        self.assertIn('id="report-data"', html)
        self.assertNotIn('<script src=', html)
        self.assertEqual(data["meta"]["dataset_kind"], "synthetic")
        self.assertEqual(set(data["meta"]["variants"]), {"raw"})
        self.assertEqual(report(self.folder), self.root / 'report.html')

    def test_core_audit_failure_blocks_old_successful_tables(self):
        audit=json.loads((self.root/'audit.json').read_text())
        audit.update(gate_passed=False,blocking=["重复价格键"])
        write_json(self.root,'audit.json',audit)
        data,_=render(self.root)
        self.assertEqual(data['factors'],[])
        self.assertIn('重复价格键',' '.join(data['issues']))

    def test_changed_config_digest_blocks_old_results(self):
        config=json.loads((self.root/'config.frozen.json').read_text())
        config['preprocess']['winsorize']=True
        write_json(self.root,'config.frozen.json',config)
        data,_=render(self.root)
        self.assertTrue(data['issues'])
        self.assertEqual(data['factors'],[])

    def test_failure_record_blocks_previous_success(self):
        write_json(self.root,'failure.json',{'message':'new calculation failed'})
        data,_=render(self.root)
        self.assertEqual(data['factors'],[])
        self.assertIn('new calculation failed',' '.join(data['issues']))

    def test_factor_output_path_cannot_escape_the_run(self):
        result=json.loads((self.root/'results.json').read_text())
        result['factors'][0]['output_dir']='../../outside'
        write_json(self.root,'results.json',result)
        data,_=render(self.root)
        self.assertFalse(any(f.get('portfolio') for f in data['factors']))
        self.assertFalse((self.root.parent/'outside'/'report.html').exists())

    def test_incomplete_panel_blocks_old_success(self):
        manifest=json.loads((self.root/'panel_manifest.json').read_text())
        manifest['status']='blocked';write_json(self.root,'panel_manifest.json',manifest)
        data,_=render(self.root)
        self.assertTrue(data['issues']);self.assertEqual(data['factors'],[])


class UnifiedReportValueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        signals,labels,benchmark,weeks=_fixture(first_low=.01,first_high=.1)
        other_signals,other_labels=signals.copy(),labels.copy()
        other_signals['code']='1'+other_signals.code.str[1:]
        other_labels['code']='1'+other_labels.code.str[1:]
        signals=pd.concat([signals,other_signals],ignore_index=True)
        labels=pd.concat([labels,other_labels],ignore_index=True)
        signals.loc[signals.group.eq(10),'group']=9
        labels.loc[0,['return','unresolved']]=[np.nan,True]
        self.tables=evaluate_factor(signals,labels,benchmark,weeks,missing_returns='available_case',empty_groups='cash')
        create_run(self.root,tables=self.tables)

    def test_partial_labels_and_cash_are_preserved_exactly(self):
        data,_=render(self.root)
        rows=pd.DataFrame(data['factors'][0]['portfolio'])
        low=rows[rows.portfolio.eq('Q1')].sort_values('entry')
        high=rows[rows.portfolio.eq('Q10')].sort_values('entry')
        self.assertAlmostEqual(low.iloc[0]['return'],.01)
        self.assertAlmostEqual(low.iloc[0]['missing_weight'],.5)
        self.assertFalse(low.iloc[0]['data_complete'])
        self.assertTrue(high['return'].eq(0).all())
        self.assertTrue(high.cash_weight.eq(1).all())
        self.assertAlmostEqual(low.nav.iloc[-1],1.01*1.5*1.5)

    def test_saved_performance_is_preserved_not_reestimated(self):
        data,_=render(self.root)
        got=pd.DataFrame(data['factors'][0]['performance'])
        want=self.tables['performance']
        for portfolio in ['Q1','Q10','Q10-Q1']:
            a=got.loc[got.portfolio.eq(portfolio)].iloc[0]
            b=want.loc[want.portfolio.eq(portfolio)].iloc[0]
            self.assertAlmostEqual(a.total_return,b.total_return)
            self.assertAlmostEqual(a.max_drawdown,b.max_drawdown)

    def test_missing_periods_are_null_not_zero(self):
        signals,labels,benchmark,weeks=_fixture(first_low=.01,first_high=.1)
        labels.loc[0,['return','unresolved']]=[np.nan,True]
        tables=evaluate_factor(signals,labels,benchmark,weeks,missing_returns='report')
        create_run(self.root,tables=tables)
        data,_=render(self.root)
        rows=[x for x in data['factors'][0]['portfolio'] if x['portfolio']=='Q1']
        self.assertIsNone(rows[0]['return'])
        perf=next(x for x in data['factors'][0]['performance'] if x['portfolio']=='Q1')
        self.assertIsNone(perf['total_return'])
        self.assertIsNone(perf['max_drawdown'])


class UnifiedReportContractTests(unittest.TestCase):
    def test_large_offline_report_preserves_the_full_json_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_run(root)
            # Include Unicode, nulls and tag-like source text, as real exports do.
            source = {"factors": [], "industry_mapping": [
                {"industry_name": "银行</script>\u2028\u2029", "weight": None}],
                "notes": "原始信号字段记录" * 150_000}
            with patch("fundamental_factor.report.build_report_data", return_value=source):
                path = report(root)
            document = path.read_text(encoding="utf-8")
            embedded = json.loads(re.search(
                r'<script id="report-data"[^>]*>(.*?)</script>', document, re.S).group(1))
            self.assertEqual(embedded["encoding"], "gzip-base64")
            decoded = gzip.decompress(base64.b64decode(embedded["data"])).decode("utf-8")
            downloaded = (root / "report_data.json").read_text(encoding="utf-8")
            self.assertEqual(decoded, downloaded)
            self.assertEqual(json.loads(decoded), source)

    def test_monthly_frequency_and_units_come_from_frozen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=settings()
            config['backtest']['rebalance_frequency']='monthly'
            config['metrics']={'annualization':12,'nw_lag':3}
            create_run(root,config);data,_=render(root)
            self.assertEqual(data['meta']['frequency'],'monthly')
            self.assertEqual(data['meta']['annualization'],12)
            self.assertEqual(data['meta']['period_label'],'月')
            self.assertEqual(data['meta']['sample_policy'],'independent')

    def test_processing_settings_are_available_for_correct_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=settings()
            config['preprocess']={'winsorize':True,'standardize':False}
            config['smoothing']={'method':'ema','span':4,'min_periods':1}
            create_run(root,config);data,_=render(root)
            self.assertEqual(data['config']['preprocess'],config['preprocess'])
            self.assertEqual(data['config']['smoothing'],config['smoothing'])

    def test_independent_coverage_does_not_become_common_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=settings();config['neutralization']['variants']=['raw','industry','full']
            create_run(root,config)
            pd.DataFrame({'entry':['2025-02-05'],'factor':['ROIC_new'],'raw_coverage':[.8],
                          'common_coverage':[.5],'variant_raw_coverage':[.8],
                          'variant_industry_coverage':[.7],'variant_full_coverage':[.5]}).to_csv(root/'coverage.csv',index=False)
            data,_=render(root);row=data['factors'][0]['coverage'][0]
            self.assertEqual(row['variant_raw_coverage'],.8)
            self.assertEqual(row['variant_industry_coverage'],.7)
            self.assertEqual(row['variant_full_coverage'],.5)
            self.assertEqual(row['common_coverage'],.5)
            self.assertEqual(data['meta']['sample_policy'],'independent')

    def test_market_cap_proxy_contract_is_not_replaced_by_actual_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=settings()
            config['neutralization']['variants']=['raw','size']
            config['neutralization']['market_cap']={'method':'quote_proxy','volume_unit':'lot','turnover_unit':'percent','price_basis':'unadjusted'}
            create_run(root,config);data,_=render(root)
            self.assertEqual(data['config']['neutralization']['market_cap'],config['neutralization']['market_cap'])
            self.assertFalse(data['profiles'].get('ROIC_new'))

    def test_actual_cap_source_and_independent_size_coverage_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=settings();config['neutralization']['variants']=['raw','size']
            config['neutralization']['market_cap']={'method':'history','source':'float_mcap','column':'float_value_yuan'}
            create_run(root,config)
            pd.DataFrame({'entry':['2025-02-05'],'factor':['ROIC_new'],'variant_raw_coverage':[1.],
                          'variant_size_coverage':[.7]}).to_csv(root/'coverage.csv',index=False)
            data,_=render(root);row=data['factors'][0]['coverage'][0]
            self.assertEqual(data['config']['neutralization']['market_cap']['column'],'float_value_yuan')
            self.assertEqual(row['variant_raw_coverage'],1.)
            self.assertEqual(row['variant_size_coverage'],.7)

    def test_sharpe_and_turnover_values_are_not_reinterpreted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tables=evaluate_factor(*turnover_fixture())
            create_run(root,tables=tables);data,_=render(root)
            got=pd.DataFrame(data['factors'][0]['performance']).set_index('portfolio')
            want=tables['performance'].set_index('portfolio')
            for portfolio in ['Q10','Q10-Q1']:
                for field in ['sharpe_ratio','excess_sharpe','mean_turnover_one_way','p95_turnover_one_way']:
                    self.assertAlmostEqual(got.loc[portfolio,field],want.loc[portfolio,field])
            self.assertIn('target_turnover_one_way',data['factors'][0]['portfolio'][0])
            self.assertIn('initial_build',data['factors'][0]['portfolio'][0])

    def test_legacy_metrics_are_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tables=evaluate_factor(*turnover_fixture())
            tables['performance']=tables['performance'].drop(columns=['sharpe_ratio','excess_sharpe'])
            create_run(root,tables=tables);data,_=render(root)
            self.assertNotIn('sharpe_ratio',data['factors'][0]['performance'][0])
            self.assertNotIn('excess_sharpe',data['factors'][0]['performance'][0])

    def test_industry_mapping_and_audit_dates_are_used_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=settings();config['industry']={'classification':'aiq_zx35','column':'industry'}
            create_run(root,config,audit={'sources':{'industry':{'start':'2022-02-03','end':'2025-06-30'}}})
            pd.DataFrame({'class':[1,24,35],'industry_code':['AIQ_ZX_01','AIQ_ZX_24','AIQ_ZX_35'],
                          'industry_name':['银行','化学制药','综合']}).to_csv(root/'industry_mapping.csv',index=False)
            data,_=render(root)
            self.assertEqual([r['industry_name'] for r in data['industry_mapping']],['银行','化学制药','综合'])
            self.assertEqual(data['audit']['sources']['industry']['start'],'2022-02-03')
            self.assertEqual(data['meta']['industry_classification'],'aiq_zx35')
