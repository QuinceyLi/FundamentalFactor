"""Summarise completed local AIQ validation and periodic factor-test results."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from fundamental_factor.data import asof_snapshot, fingerprint, read_frame, write_json


def md_table(frame: pd.DataFrame) -> str:
    def text(value):
        if pd.isna(value): return "—"
        if isinstance(value, float): return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join("---" for _ in frame.columns) + " |"] +
                      ["| " + " | ".join(map(text, row)) + " |" for row in frame.itertuples(index=False, name=None)])


def generate(prepared: Path, runs: list[Path], output: Path, baseline: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((prepared/"manifest.json").read_text())
    protected = inputs["price_source_protected"]
    unchanged = fingerprint(Path(protected["path"])) == protected
    if not unchanged:
        raise ValueError("Protected repaired price source changed")
    sections, md, validation = [], [], {"repaired_price_unchanged": unchanged, "runs": []}
    def paragraph(text):
        sections.append("<p>" + html.escape(text) + "</p>");md.append(text)
    def title(text):
        sections.append("<h2>" + html.escape(text) + "</h2>");md.append("## " + text)
    def table(frame):
        display = frame.copy()
        for column in ["比例", "一致率", "有效行业比例", "基础池ST未知比例"]:
            if column in display:
                display[column] = display[column].map(lambda value: f"{value:.6%}" if pd.notna(value) else "—")
        sections.append('<div class="scroll">' + display.to_html(index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.6g}") + '</div>')
        md.append(md_table(display))
    title("AIQ 数据接入核验与因子测试结果")
    paragraph("核验日期：2026-10-28。已使用AIQ历史中信行业、ST/简称、直接流通市值、证券生命周期、交易日历和原始量价代理。ROIC与公募持仓变化成品仍来自Mongo明细；收益仍来自已修复quotefac_repaired.feather，指纹核对未改变。")
    paragraph("主测试为周度与月度，分别比较raw、行业、市值、行业市值中性化，使用独立样本；EMA、去极值、标准化关闭。无模型训练或持仓优化。原始因子的样本不依赖行业/市值/Barra完整性。")
    title("当前数据选择")
    table(pd.DataFrame(inputs["source_selection"].items(), columns=["入口", "实际来源与优先级"]))
    title("行业对应方式")
    paragraph("严格采用FACFUNC.get_industry → Tools/Toolbox.zxclassified() → get_classname() 的35类及中文名称。原始中信一二三级字段另外保留。class24叫化学制药但实际包含全部一级医药；class25/26分别为太阳能及光伏/新能源动力系统；class35为混合综合类。旧Excel标签不用于命名。")
    from fundamental_factor.aiq_industry import industry_catalog
    table(industry_catalog()[["class", "name", "industry_code", "rule"]])
    title("接入核验与异常处理")
    cap = inputs["datasets"]["market_cap"]
    st = inputs["datasets"]["st_combined"]
    status = inputs["datasets"]["status"]
    paragraph("行业表已导出5804只股票、9516个有效/空值事件；终止日按包含当日、次日清空。股本与市值采用日期向后匹配，均不跳过显式坏值。行业规则和生效区间不具有完整首次发布/修订档案，相关假设保留披露。")
    paragraph(f"市值原始{cap['input_stats']['market_feature']['source_rows']:,}行，过滤非交易日{cap['input_stats']['market_feature']['non_trading_date']:,}行，保留{cap['counts']['output_rows']:,}个股票日；{cap['statuses'].get('nonfinite_or_nonpositive',0):,}条无效市值以NaN事件保留。单位万元×10000转元。普通核验分歧不改源值，不在没有共同证据时删除有效市值。")
    totals = cap["comparison_totals"]
    table(pd.DataFrame([
        {"核验": "股本×原始收盘", "可比记录": totals["share_compared"], "2%内记录": totals["share_within_tolerance"], "一致率": totals["share_within_tolerance"]/totals["share_compared"]},
        {"核验": "同日量价/换手率代理", "可比记录": totals["proxy_compared"], "2%内记录": totals["proxy_within_tolerance"], "一致率": totals["proxy_within_tolerance"]/totals["proxy_compared"]}]))
    paragraph(f"AIQ简称历史覆盖代理上限为2022-11-25，之后不延续旧名。AIQ未知由官方SZSE历史补充；{st['conflicting_boundary_rows']}个ST边界分歧由官方事件覆盖并留痕。停牌仍为量价代理，与既有同日代理分歧{status['raw_proxy_disagreements']}条；{status['suspension_events_with_nonzero_volume_close']:,}条停牌事件所在日仍有非零量价，单列核验，没有直接当作全天停牌。")
    title("实际样本、覆盖与缺失")
    coverage_rows, ic_rows = [], []
    for run in runs:
        manifest = json.loads((run/"panel_manifest.json").read_text())
        result = json.loads((run/"results.json").read_text())
        if result["status"] != "completed": raise ValueError(f"Run incomplete: {run}")
        config = json.loads((run/"config.frozen.json").read_text())
        if Path(config["sources"]["quotes"]) != Path(protected["path"]): raise ValueError("Unexpected return source")
        frequency = config["backtest"]["rebalance_frequency"]
        relative = Path(__import__('os').path.relpath(run/"report.html", output)).as_posix()
        sections.append(f'<p><a href="{html.escape(relative)}">{html.escape(frequency)} 完整因子报告</a></p>')
        md.append(f"[{frequency} 完整因子报告]({relative})")
        quality = pd.read_csv(run/"data_quality_summary.csv")
        unresolved_delist = int(quality.set_index("指标").loc["退市退出未解决", "记录数"])
        summary = {"调仓": frequency, "完整期数": manifest["periods"], "入场起": str(manifest["holding_start"])[:10],
                   "最后退出": str(manifest["holding_end"])[:10], "基础股期次": manifest["base_rows"],
                   "基础池ST未知比例": manifest["unknown_st_base_rows"] / manifest["base_rows"],
                   "未解决收益": manifest["unresolved_labels"], "未解决退市": unresolved_delist}
        coverage_rows.append(summary);validation["runs"].append(summary)
        title(f"{frequency} 的来源与缺失分母")
        table(quality)
        if frequency == "weekly":
            table(pd.read_csv(run/"data_usage.csv"))
            title("周度实际行业与原始中信一级对应")
            table(pd.read_csv(run/"industry_distribution.csv"))
            controls = pd.read_parquet(run/"controls.parquet")
            missing = controls.loc[controls.industry.isna() | ~controls.market_cap_available]
            selected = [c for c in ["entry", "code", "industry", "citic_level1_name", "industry_status", "industry_source_date",
                "float_mcap", "market_cap_date", "market_cap_invalid_reason", "st_source"] if c in missing]
            missing[selected].to_csv(output/"missing_controls_weekly.csv",index=False)
            paragraph(f"周度基础股票池共{len(controls):,}个股周次，行业缺失{int(controls.industry.isna().sum())}条，市值缺失{int((~controls.market_cap_available).sum())}条。逐条记录另存missing_controls_weekly.csv，不用旧值覆盖。")
        for factor in result["factors"]:
            metrics = pd.read_csv(run/factor["output_dir"]/"ic_summary.csv")
            selected = metrics.loc[metrics.method.eq("spearman") & metrics.subset.eq("ALL"), ["variant", "ic_mean", "annual_icir", "n"]].copy()
            selected.insert(0,"因子",factor["name"]);selected.insert(0,"调仓",frequency)
            ic_rows.append(selected)
    title("运行概况")
    table(pd.DataFrame(coverage_rows))
    title("本轮各版本 RankIC")
    paragraph("本轮结果是使用当前数据口径的描述性因子检验；版本间样本可能不同，不能把差异全部归因于中性化。多空/多头超额夏普、净值、换手率和完整分组表在各频率HTML报告中。")
    table(pd.concat(ic_rows, ignore_index=True))
    if baseline is not None and (baseline/"config.frozen.json").is_file():
        old_config = json.loads((baseline/"config.frozen.json").read_text())
        old_industry = old_config.get("industry", {})
        old_path = Path(old_config["sources"].get(old_industry.get("source", "industry"), ""))
        current_weekly = next((p for p in runs if "weekly" in p.name), None)
        if current_weekly and old_path.is_file():
            new = pd.read_parquet(current_weekly/"controls.parquet", columns=["cutoff", "code", "industry"])
            field = old_industry.get("column", "industry")
            old = read_frame(old_path, ["date", "code", field]).rename(columns={field: "old_industry"})
            compared = asof_snapshot(new[["cutoff", "code"]], old, "old_industry")
            title("与此前行业源覆盖的比较")
            table(pd.DataFrame([{"来源": "此前中信一级30类/记录更新时间代理", "基础股周次":len(compared),"有效行业比例":float(compared.old_industry.notna().mean())},
                                {"来源": "本轮AIQ历史区间/35类规则", "基础股周次":len(new),"有效行业比例":float(new.industry.notna().mean())}]))
            paragraph("本表将两个行业源都向后匹配到本轮同一基础股票池截点，分母相同。行业体系和可用日期假设不同；只比较可取得分类的比例，不作为收益改善或历史信息发布时间已认证的证据。")
    title("仍可完善的数据")
    for item in inputs["remaining_improvements"]: paragraph(item)
    title("验证与复现")
    acceptance = output / "actual_acceptance.json"
    if acceptance.is_file():
        actual_checks = json.loads(acceptance.read_text())
        table(pd.DataFrame(actual_checks.get("runs", [])))
        validation["actual_acceptance"] = actual_checks
        sections.append('<p><a href="actual_acceptance.json">下载实际面板验收记录</a></p>')
        md.append("[实际面板验收记录](actual_acceptance.json)")
    logs = [prepared.parent/"aiq_unit_tests_final.log", prepared.parent/"aiq_custom_cap_column_test.log"]
    test_rows = []
    for path in logs:
        text = path.read_text() if path.is_file() else ""
        matches = re.findall(r"Ran (\d+) tests? in ([\d.]+)s", text)
        test_rows.append({"日志": path.name, "测试数": int(matches[-1][0]) if matches else 0,
                          "通过": bool(matches and text.rstrip().endswith("OK"))})
    validation["test_runs"] = test_rows
    table(pd.DataFrame(test_rows))
    paragraph("测试包含日期精度、信号截点、过期清空、来源冲突、无效最新值、原始样本独立性及完整6周合成回测。源数据未修改；准备脚本前后与回测后均核对已修复价格指纹。")
    paragraph("配置：configs/factor_test_aiq_weekly.json、factor_test_aiq_monthly.json。准备：python3 scripts/prepare_aiq_inputs.py。研究：python3 -m fundamental_factor run --config <配置>。接入细节和35类名称见 docs/aiq_data_integration.md；本轮所有输入和规则指纹见 outputs/aiq_inputs_20261028/manifest.json。")
    style='body{font:15px/1.8 system-ui,sans-serif;color:#253545;background:#f3f6f9;margin:0}main{max-width:1320px;margin:24px auto;padding:30px;background:white}h2{color:#1b537a;margin-top:32px;border-bottom:1px solid #d9e4ed}table{border-collapse:collapse;width:100%;font-size:13px;overflow-wrap:anywhere}td,th{padding:8px;border:1px solid #dbe3e9;text-align:left}th{background:#edf4fa}.scroll{overflow:auto}a{color:#176592}p{max-width:1200px}@media print{main{margin:0;padding:0}.scroll{overflow:visible}}'
    (output/"report.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AIQ接入核验与因子测试</title><style>'+style+'</style><main>'+"\n".join(sections)+'</main></html>')
    (output/"AIQ接入核验报告.md").write_text("\n\n".join(md)+"\n")
    pd.DataFrame(coverage_rows).to_csv(output/"run_summary.csv",index=False)
    pd.concat(ic_rows,ignore_index=True).to_csv(output/"ic_comparison.csv",index=False)
    write_json(output/"validation.json",validation)
    print(output/"report.html")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("outputs/aiq_inputs_20261028"))
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=Path("outputs/aiq_integration_validation_20261028"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/diagnostics_validation_20261028/weekly_none"))
    args = parser.parse_args()
    runs = args.run_dir or [Path("outputs/factor_test_aiq_weekly"),Path("outputs/factor_test_aiq_monthly")]
    generate(args.prepared.resolve(), [p.resolve() for p in runs], args.output.resolve(), args.baseline.resolve())
