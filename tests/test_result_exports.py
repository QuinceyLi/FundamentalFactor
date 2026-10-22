import tempfile
import unittest
from pathlib import Path
import pandas as pd
from fundamental_factor.research import save_result_tables


class ResultExportTests(unittest.TestCase):
    def test_large_ledger_removes_interrupted_csv_and_preserves_every_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "holdings.csv").write_text("incomplete old export")
            frame = pd.DataFrame({"code": ["000001.SZ"] * 4, "weight": [.1, .2, .3, .4]})
            result = save_result_tables(path, {"holdings": frame, "summary": frame.iloc[:1]}, holdings_csv_limit=3)
            self.assertFalse((path / "holdings.csv").exists())
            pd.testing.assert_frame_equal(pd.read_parquet(path / "holdings.parquet"), frame)
            self.assertTrue((path / "summary.csv").exists())
            self.assertEqual(result["holdings"], {"rows": 4, "formats": ["parquet"]})

    def test_small_ledger_has_both_download_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            result = save_result_tables(path, {"holdings": pd.DataFrame({"weight": [.5, .5]})}, holdings_csv_limit=2)
            self.assertEqual(result["holdings"]["formats"], ["csv", "parquet"])
            self.assertEqual(pd.read_csv(path / "holdings.csv").weight.sum(), 1)
