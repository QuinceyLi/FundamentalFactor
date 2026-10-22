from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from fundamental_factor.config import load_config
from fundamental_factor.report import report
from fundamental_factor.research import run_research, unresolved_held_delistings
from tests.fixtures import disk_fixture, enable_execution_fixture


class ScopeAndTerminalTests(unittest.TestCase):
    def test_industry_and_exchange_scope_reach_both_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            config = enable_execution_fixture(disk_fixture(Path(folder), weeks_count=3))
            raw = json.loads(config.path.read_text())
            raw['neutralization'] = {'include_float_mcap': False}
            raw['universe'] = {'exchanges': ['SH', 'SZ']}
            config.path.write_text(json.dumps(raw))
            pd.read_parquet(config.source('history')).drop(columns='float_mcap').to_parquet(config.source('history'))
            config = load_config(config.path)
            result = run_research(config)
            report(config.output)
            self.assertEqual(result['benchmark']['exchanges'], ['SH', 'SZ'])
            for factor in result['factors']:
                body = (config.output / factor['output_dir'] / 'report.html').read_text()
                self.assertIn('纯行业中性仅使用行业标签', body)
                self.assertIn('投资域为沪深 A 股', body)
                self.assertIn('行业中性化信号：执行账户', body)
                self.assertNotIn('行业市值中性化信号', body)
                self.assertNotIn('执行账本产物尚不完整', body)

    def test_stock_sold_before_delisting_needs_no_cash_settlement(self):
        dates = pd.to_datetime(['2025-01-13', '2025-01-20', '2025-01-20'])
        holdings = pd.DataFrame({'code': ['000001.SZ', '000001.SZ', '000002.SZ'],
                                 'shares': [100., 0., 50.], 'exit': dates})
        universe = pd.DataFrame({'code': ['000001.SZ', '000002.SZ'],
                                'delisting_date': pd.to_datetime(['2025-01-15', '2025-01-15'])})
        self.assertEqual(unresolved_held_delistings(holdings, universe, set()), {'000002.SZ'})
        self.assertEqual(unresolved_held_delistings(holdings, universe, {'000002.SZ'}), set())


if __name__ == '__main__':
    unittest.main()
