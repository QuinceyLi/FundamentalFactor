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
    def test_industry_exchange_scope_and_execution_reach_unified_report(self):
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
            data = json.loads((config.output / 'report_data.json').read_text())
            self.assertEqual(data['config']['universe']['exchanges'], ['SH', 'SZ'])
            self.assertFalse(data['config']['neutralization']['include_float_mcap'])
            self.assertIn('industry', data['meta']['variants'])
            self.assertNotIn('industry_size', data['meta']['variants'])
            for factor in data['factors']:
                self.assertTrue(factor['execution']['weekly'])
                self.assertFalse(factor['execution']['issues'])

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
