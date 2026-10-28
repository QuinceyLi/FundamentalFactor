"""Reports disclose smoothing, industry provenance and the added diagnostics."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from fundamental_factor.backtest import evaluate_factor
from fundamental_factor.report import (_Charts, _definitions, _industry_disclosure,
                                       _turnover_section, _variant_labels, report)
from tests.test_turnover_sharpe import fixture


def config():
    return {"schema_version": 2, "dataset_kind": "synthetic",
            "factors": [{"name": "Example", "label": "测试因子"}],
            "preprocess": {"winsorize": False, "standardize": False},
            "smoothing": {"method": "ema", "span": 4, "min_periods": 1},
            "neutralization": {"variants": ["raw"], "sample_policy": "independent"},
            "industry": {"classification": "citic_l1", "source": "industry", "column": "industry_name"},
            "metrics": {"annualization": 52, "nw_lag": 13, "turnover_warning_threshold": .5},
            "data_policy": {"mode": "classic", "missing_returns": "available_case", "empty_groups": "cash"}}


class TurnoverSharpeReportTests(unittest.TestCase):
    def test_ema_labels_and_definitions_do_not_claim_unmodified_inputs(self):
        settings = config()
        body = _definitions(settings)
        self.assertEqual(_variant_labels(settings)["raw"], "未中性化平滑信号")
        self.assertIn("span=4、min_periods=1", body)
        self.assertIn("股票池出现断档时重置状态", body)
        self.assertIn("value 保留", body)
        self.assertNotIn("signal=value", body)
        settings["smoothing"]["method"] = "none"
        self.assertEqual(_variant_labels(settings)["raw"], "原样输入信号")
        body = _definitions(settings)
        self.assertIn("signal=value", body)
        self.assertIn("不判断外部已经计算好的因子是否自带平滑", body)

    def test_sharpe_and_turnover_definitions_are_explicit_and_configurable(self):
        settings = config()
        settings["metrics"]["turnover_warning_threshold"] = .75
        body = _definitions(settings)
        for text in ["算术期均值/期收益样本标准差", "基准在双腿抵消，不再扣一次",
                     "几何年化收益/年化波动率", "75.00%", "排除每腿首次实际建仓",
                     "可配置诊断而非通用有效性标准", "没有再除以2单位毛敞口", "Empyrical sharpe_ratio"]:
            self.assertIn(text, body)

    def test_industry_coverage_uses_audited_availability_not_a_hardcoded_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({"industry_code": ["40"], "industry_name": ["银行"]}).to_csv(
                root / "industry_mapping.csv", index=False)
            checked = {"sources": {"industry": {"start": "2022-02-03", "end": "2025-06-30"}}}
            body = _industry_disclosure(root, config(), checked, prefix="../../")
            self.assertIn("2022-02-03 至 2025-06-30", body)
            self.assertIn("更早日期和未映射记录保持缺失", body)
            self.assertIn('href="../../industry_mapping.csv"', body)
            self.assertIn("银行", body)
            self.assertNotIn("2021-07", body)

    def test_turnover_charts_and_diagnostics_include_both_weight_definitions(self):
        tables = evaluate_factor(*fixture())
        body = _turnover_section(tables["portfolio_weekly"], tables["performance"], _Charts(), config())
        for text in ["逐期单边漂移换手率", "逐期目标单边换手率", "持续换手缺失期数",
                     "首次建仓期数", "配置警戒值", "50.00%", "缺失不当作0或低换手"]:
            self.assertIn(text, body)

    def test_complete_report_shows_long_short_summary_and_keeps_legacy_results_readable(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = root / "factors" / "Example"
                folder.mkdir(parents=True)
                settings = config()
                digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                records = {
                    "config.frozen.json": settings,
                    "audit.json": {"gate_passed": True, "config_digest": digest, "sources": {}},
                    "panel_manifest.json": {"status": "ready", "config_digest": digest},
                    "results.json": {"schema_version": 2, "dataset_kind": "synthetic", "status": "completed",
                                     "config_digest": digest, "factors": [{"name": "Example", "output_dir": "factors/Example"}]},
                }
                for filename, value in records.items():
                    (root / filename).write_text(json.dumps(value, ensure_ascii=False))
                tables = evaluate_factor(*fixture())
                if legacy:
                    extras = {"sharpe_ratio", "sharpe_n_periods", "excess_sharpe", "excess_sharpe_n_periods",
                              "turnover_one_way", "target_turnover_one_way", "initial_build", "high_turnover",
                              "initial_build_periods", "mean_turnover_one_way", "p95_turnover_one_way",
                              "high_turnover_fraction", "turnover_observations", "turnover_missing_periods",
                              "turnover_warning_threshold"}
                    tables = {name: frame.drop(columns=list(set(frame) & extras)) for name, frame in tables.items()}
                for name, frame in tables.items():
                    frame.to_csv(folder / f"{name}.csv", index=False)
                report(root)
                body = (folder / "report.html").read_text()
                self.assertIn("未中性化平滑信号", body)
                self.assertIn("十分组组合表现", body)
                if not legacy:
                    for text in ["多头与多空超额夏普", "多头（Q10−基准）", "多空（Q10−Q1）",
                                 "超额夏普有效期数", "换手率与高换手诊断", "年度分组与多空表现"]:
                        self.assertIn(text, body)
                self.assertNotIn("<script src=", body)


if __name__ == "__main__":
    unittest.main()
