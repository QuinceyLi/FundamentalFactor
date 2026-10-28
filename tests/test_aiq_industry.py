import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fundamental_factor.aiq_industry import (
    CLASSIFICATION, build_aiq_industry_history, classify_aiq_industry,
    industry_catalog, normalize_industry, prepare_aiq_industry_history,
)
from fundamental_factor.data import asof_snapshot


def records(rows):
    return pd.DataFrame(rows, columns=["code", "induscode", "industry", "start", "end"])


def levels(first=26, second=2620, third=262010, *, code="000001.SZ", start=20200101, end=None):
    return {i: records([(code, number, f"原始层{i}名称", start, end)])
            for i, number in enumerate((first, second, third), 1)}


def at(history, days, code="000001.SZ"):
    targets = pd.DataFrame({"code": [code] * len(days), "cutoff": [pd.Timestamp(day) for day in days]})
    return asof_snapshot(targets, history, "aiq")


class AIQIndustryTests(unittest.TestCase):
    def test_all_thirty_five_classes_and_source_name_quirks(self):
        fixtures = [
            (40, 4010, 401010, 1), (41, 4110, 411010, 2), (41, 4120, 412010, 3),
            (42, 4210, 421010, 4), (10, 1010, 101010, 5), (11, 1110, 111010, 6),
            (12, 1230, 123010, 7), (12, 1210, 121010, 8), (20, 2020, 202010, 9),
            (20, 2010, 201010, 10), (21, 2110, 211010, 11), (22, 2210, 221010, 12),
            (23, 2310, 231010, 13), (24, 2410, 241010, 14), (25, 2510, 251010, 15),
            (26, 2620, 262010, 16), (27, 2710, 271010, 17), (28, 2810, 281010, 18),
            (30, 3010, 301010, 19), (31, 3110, 311010, 20), (32, 3210, 321010, 21),
            (33, 3310, 331010, 22), (34, 3410, 341010, 23), (35, 3520, 352010, 24),
            (26, 2620, 262060, 25), (27, 2740, 274010, 26), (36, 3610, 361010, 27),
            (36, 3620, 362010, 28), (37, 3710, 371010, 29), (50, 5010, 501010, 30),
            (60, 6010, 601010, 31), (61, 6110, 611010, 32), (62, 6210, 621010, 33),
            (63, 6310, 631010, 34), (70, 7010, 701010, 35),
        ]
        frame = pd.DataFrame(fixtures, columns=["zx1code", "zx2code", "zx3code", "expected"])
        frame["code"] = "000001.SZ"
        actual = classify_aiq_industry(frame)
        self.assertEqual(actual.custom_class_code.tolist(), list(range(1, 36)))
        catalog = industry_catalog()
        self.assertEqual(len(catalog), 35)
        self.assertEqual(catalog.industry_code.nunique(), 35)
        self.assertEqual(actual.custom_class_name.iloc[23], "化学制药")
        self.assertEqual(actual.custom_class_name.iloc[24:26].tolist(), ["太阳能及光伏", "新能源动力系统"])
        self.assertTrue(catalog.classification.eq(CLASSIFICATION).all())

    def test_combined_class_35_and_conditional_ticker_overrides(self):
        frame = pd.DataFrame([
            ("600739.SH", 41, 4110, 411010), ("600733.SH", 41, 4110, 411010),
            ("600291.SH", 41, 4120, 412010), ("000623.SZ", 41, 4110, 411010),
            ("000712.SZ", 41, 4110, 411010), ("600739.SH", 40, 4010, 401010),
            ("000001.SZ", 41, 4130, 413010), ("000001.SZ", 42, 4220, 422010),
            ("000001.SZ", 43, 4310, 431010), ("000001.SZ", 70, 7010, 701010),
            ("000001.SZ", 27, 2730, 273030),
        ], columns=["code", "zx1code", "zx2code", "zx3code"])
        actual = classify_aiq_industry(frame)
        self.assertEqual(actual.custom_class_code.tolist(), [35] * 5 + [1] + [35] * 4 + [25])
        self.assertEqual(actual.industry_ticker_override.tolist(), [True] * 5 + [False] * 6)

    def test_normalization_keeps_namespace_separate_from_citic(self):
        result = normalize_industry(pd.Series([" AIQ_ZX_01 ", "aiq_zx_25", "化学制药", "医药", "35", "CI005021", None, ""]))
        self.assertEqual(result.industry_name.iloc[:3].tolist(), ["银行", "太阳能及光伏", "化学制药"])
        self.assertTrue(result.industry_name.iloc[3:].isna().all())
        self.assertEqual(result.industry_mapping_status.tolist(), ["mapped"] * 3 + ["unmapped"] * 3 + ["missing"] * 2)

    def test_no_primary_or_required_sublevel_is_unknown_not_general_class(self):
        frame = pd.DataFrame([
            (None, 4110, 411010), (41, None, None), (42, 4210, None),
            (12, None, None), (20, None, None), (26, None, None),
            (27, None, 273030), (36, 3610, None), (40, None, None),
        ], columns=["zx1code", "zx2code", "zx3code"])
        frame["code"] = "000001.SZ"
        actual = classify_aiq_industry(frame)
        self.assertTrue(actual.custom_class_code.iloc[:8].isna().all())
        self.assertEqual(actual.custom_class_code.iloc[8], 1)

    def test_no_lookahead_and_explicit_end_event_prevents_forward_carry(self):
        history, _, _ = build_aiq_industry_history(levels(start=20200103, end=20200105))
        observed = at(history, ["2020-01-02 23:59:59", "2020-01-03", "2020-01-05 23:59:59", "2020-01-06", "2020-02-01"])
        self.assertTrue(pd.isna(observed.industry.iloc[0]))
        self.assertEqual(observed.industry.iloc[1:3].tolist(), ["机械", "机械"])
        self.assertTrue(observed.industry.iloc[3:].isna().all())
        self.assertEqual(history.date.tolist(), [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")])
        self.assertEqual(observed.citic_level1_name.iloc[1], "原始层1名称")

    def test_exclusive_end_is_supported_without_shifting_start(self):
        history, _, _ = build_aiq_industry_history(levels(start=20200103, end=20200105), end_inclusive=False)
        observed = at(history, ["2020-01-04", "2020-01-05"])
        self.assertEqual(observed.industry.iloc[0], "机械")
        self.assertTrue(pd.isna(observed.industry.iloc[1]))

    def test_async_sublevel_change_is_applied_without_matching_level1_start(self):
        source = levels()
        source[3] = records([
            ("000001.SZ", 262010, "普通机械", 20200101, 20200103),
            ("000001.SZ", 262060, "光伏设备", 20200104, None),
        ])
        history, _, _ = build_aiq_industry_history(source)
        observed = at(history, ["2020-01-03 23:59:59", "2020-01-04"])
        self.assertEqual(observed.industry.tolist(), ["机械", "太阳能及光伏"])
        self.assertTrue(observed.citic_level1_name.eq("原始层1名称").all())

    def test_expired_required_sublevel_remains_unknown_even_when_level1_continues(self):
        source = levels()
        source[3].loc[0, "end"] = 20200105
        history, _, _ = build_aiq_industry_history(source)
        observed = at(history, ["2020-01-05", "2020-01-06"])
        self.assertEqual(observed.industry.iloc[0], "机械")
        self.assertTrue(pd.isna(observed.industry.iloc[1]))
        self.assertEqual(observed.industry_status.iloc[1], "missing_required_level3")
        self.assertEqual(observed.citic_level1_code.iloc[1], "26")

    def test_overlap_is_unknown_only_for_affected_interval_and_disclosed(self):
        source = levels()
        source[3] = records([
            ("000001.SZ", 262010, "普通机械", 20200101, 20200105),
            ("000001.SZ", 262060, "光伏设备", 20200104, None),
        ])
        history, quarantine, _ = build_aiq_industry_history(source)
        observed = at(history, ["2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"])
        self.assertEqual(observed.industry.iloc[0], "机械")
        self.assertTrue(observed.industry.iloc[1:3].isna().all())
        self.assertEqual(observed.industry.iloc[3], "太阳能及光伏")
        self.assertEqual(quarantine.reason.tolist(), ["overlap_level3"])

    def test_hierarchy_conflict_does_not_silently_relabel_level1(self):
        source = levels(first=40, second=4110, third=411010)
        history, quarantine, _ = build_aiq_industry_history(source)
        self.assertTrue(history.industry.isna().all())
        self.assertTrue(history.industry_status.eq("hierarchy_mismatch").all())
        self.assertEqual(quarantine.reason.tolist(), ["hierarchy_mismatch"])

    def test_identical_duplicates_are_collapsed_but_invalid_interval_quarantines_code(self):
        source = levels()
        source[1] = pd.concat([source[1], source[1]], ignore_index=True)
        history, _, summary = build_aiq_industry_history(source)
        self.assertEqual(summary["identical_duplicates_collapsed"]["1"], 1)
        self.assertEqual(history.industry.iloc[0], "机械")
        source[3] = pd.concat([source[3], records([("000001.SZ", 262060, "光伏设备", 20200105, 20200104)])])
        history, quarantine, summary = build_aiq_industry_history(source)
        self.assertTrue(history.industry.isna().all())
        self.assertEqual(summary["quarantined_codes"], ["000001.SZ"])
        self.assertIn("invalid_interval_order", quarantine.reason.tolist())

    def test_prepare_exports_independent_files_and_keyed_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            source = levels()
            other = levels(first=40, second=4010, third=401010, code="000002.SZ")
            before = {}
            for i in (1, 2, 3):
                frame = pd.concat([source[i], other[i]], ignore_index=True)
                if i == 3:
                    frame = frame.iloc[::-1]
                path = source_dir / f"zx_sector{i}.csv"
                frame.to_csv(path, index=False, encoding="gb18030")
                before[path] = path.read_bytes()
            output = root / "adapted" / "aiq_industry.parquet"
            summary = prepare_aiq_industry_history(source_dir, output)
            self.assertEqual(summary["comparison"]["common_keys"], 2)
            self.assertEqual(summary["comparison"]["disagreements"], 0)
            observed = pd.read_parquet(output).set_index("code")
            self.assertEqual(observed.loc["000001.SZ", "industry"], "机械")
            self.assertEqual(observed.loc["000002.SZ", "industry"], "银行")
            for suffix in ["_mapping.csv", "_quarantine.csv", "_comparison.csv", ".manifest.json"]:
                self.assertTrue((output.parent / (output.stem + suffix)).exists())
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            with self.assertRaisesRegex(ValueError, "separate"):
                prepare_aiq_industry_history(source_dir, source_dir / "industry.parquet")

    def test_invalid_noninteger_code_is_quarantined_not_comparison_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            source_dir.mkdir()
            source = levels()
            source[2]["induscode"] = 2620.5
            for i in (1, 2, 3):
                source[i].to_csv(source_dir / f"zx_sector{i}.csv", index=False, encoding="gb18030")
            result = prepare_aiq_industry_history(source_dir, root / "industry.parquet")
            self.assertEqual(result["quarantined_codes"], ["000001.SZ"])
            self.assertTrue(pd.read_parquet(root / "industry.parquet").industry.isna().all())


if __name__ == "__main__":
    unittest.main()
