from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa

from fundamental_factor.price_repair import derive_segment_patch, factor_chain_audit, repair_adjusted_table, sina_dividend_rows


class PriceRepairTests(unittest.TestCase):
    def fixture(self):
        reference = pd.DataFrame({"date": [20250102, 20250103, 20250106, 20250107],
            "code": ["000001.SZ"] * 4, "open": [10., 11., 12., 13.],
            "close": [10., 11., 12., 13.], "last": [9., 10., 11., 12.],
            "acc_adj_factor": [2., 2., 2., 2.]})
        legacy = reference.copy()
        legacy.loc[:1, "acc_adj_factor"] = 4.
        return legacy, reference

    def test_constant_rebase_preserves_returns_and_other_fields(self):
        legacy, reference = self.fixture()
        patch = derive_segment_patch(legacy, reference, "000001.SZ", 20250106)
        self.assertEqual(patch.multiplier, .5)
        frame = legacy[["date", "code"]].copy()
        for field in ("acc_open", "acc_close", "acc_high", "acc_low", "acc_vwap"):
            frame[field] = legacy.open * legacy.acc_adj_factor
        other = frame.copy()
        other["code"] = "000002.SZ"
        frame = pd.concat([frame, other], ignore_index=True)
        frame["turnover"] = np.arange(len(frame), dtype=float)
        table = pa.Table.from_pandas(frame, preserve_index=True)
        repaired, ledger = repair_adjusted_table(table, patch)
        result = repaired.to_pandas()
        self.assertEqual(len(ledger), 2)
        self.assertEqual(table.schema, repaired.schema)
        np.testing.assert_allclose(result.loc[:3, "acc_close"], reference.close * reference.acc_adj_factor)
        pd.testing.assert_frame_equal(result.iloc[4:], frame.iloc[4:])
        np.testing.assert_array_equal(result.turnover, frame.turnover)
        self.assertAlmostEqual(result.acc_close.iloc[1] / result.acc_close.iloc[0],
                               frame.acc_close.iloc[1] / frame.acc_close.iloc[0])
        for name in ("acc_open", "acc_high", "acc_low", "acc_vwap"):
            np.testing.assert_allclose(result[name] / result.acc_close, frame[name] / frame.acc_close)

    def test_no_splicing_when_raw_feeds_disagree(self):
        legacy, reference = self.fixture()
        reference.loc[2, "open"] += .01
        with self.assertRaisesRegex(ValueError, "Raw open disagreement"):
            derive_segment_patch(legacy, reference, "000001.SZ", 20250106)

    def test_nonconstant_factor_segments_require_review(self):
        legacy, reference = self.fixture()
        legacy.loc[0, "acc_adj_factor"] = 3.9
        with self.assertRaisesRegex(ValueError, "not constant"):
            derive_segment_patch(legacy, reference, "000001.SZ", 20250106)

    def test_missing_adjacent_observation_or_duplicate_key_rejects(self):
        legacy, reference = self.fixture()
        with self.assertRaises(ValueError):
            derive_segment_patch(legacy, reference.drop(index=1), "000001.SZ", 20250106)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            derive_segment_patch(pd.concat([legacy, legacy.iloc[:1]]), reference, "000001.SZ", 20250106)

    def test_true_dividend_step_is_preserved(self):
        legacy, reference = self.fixture()
        # On the event day a 1-unit dividend lowers the reference previous close.
        for frame in (legacy, reference):
            frame.loc[2, "last"] = 10.
            frame.loc[2:, "acc_adj_factor"] = 2.2
        patch = derive_segment_patch(legacy, reference, "000001.SZ", 20250106)
        self.assertAlmostEqual(patch.multiplier, .5)
        self.assertAlmostEqual(patch.reference_factor_after / patch.reference_factor_before, 1.1)

    def test_positive_extra_adjustments_are_only_diagnostics(self):
        legacy, reference = self.fixture()
        anomalous, stats = factor_chain_audit(legacy)
        self.assertEqual(stats["deviations"], 1)
        self.assertEqual(stats["negative_deviations"], 1)
        self.assertAlmostEqual(anomalous.residual.iloc[0], -.5)
        reference.loc[2:, "acc_adj_factor"] = 4.
        anomalous, stats = factor_chain_audit(reference)
        self.assertEqual(stats["negative_deviations"], 0)
        self.assertEqual(stats["deviations"], 1)
        self.assertAlmostEqual(anomalous.residual.iloc[0], 1.)

    def test_dividend_parser_keeps_per_ten_share_units_and_implementation_status(self):
        html = '<table id="sharebonus_1">'
        for status, amount in (("实施", "3.62"), ("预案", "9")):
            html += "<tr>" + "".join("<td>" + value + "</td>" for value in
                ["2025-06-05", "0", "0", amount, status, "2025-06-12", "2025-06-11", "--", "查看"]) + "</tr>"
        html += "</table>"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dividend.html"
            path.write_bytes(html.encode("gb18030"))
            result = sina_dividend_rows(path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ex_date"], 20250612)
        self.assertEqual(result[0]["cash_per_10_shares"], 3.62)


if __name__ == "__main__":
    unittest.main()
