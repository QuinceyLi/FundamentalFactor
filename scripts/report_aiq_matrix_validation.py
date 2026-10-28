"""Summarise completed AIQ-matrix runs using small CSV/JSON outputs only.

No holdings, features, controls or label panels are loaded. Source integrity
uses bounded first/last-1-MiB fingerprints, not full-file content certification.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from fundamental_factor.data import fingerprint, write_json


PROJECT = Path(__file__).resolve().parents[1]
VARIANTS = ("raw", "industry", "size", "industry_size")
VARIANT_NAMES = {"raw": "未中性化", "industry": "行业中性", "size": "市值中性", "industry_size": "行业市值中性"}
FREQUENCY_NAMES = {"weekly": "周度", "monthly": "月度"}
RULE_ROWS = [
    {"源字段": "volume / amount / return / vwap", "规则": "zero", "处理": "首个非空前仍缺失，其后缺失补0；不是实际成交或完整收益证据"},
    {"源字段": "open / high / low / turnover / free_turnover / zdtmark / fy1 / CAGR", "规则": "keep", "处理": "保留交易日网格中的缺失，不跨日期补值"},
    {"源字段": "特定日内 vwap/twap_30toe…210toe", "规则": "last", "处理": "用同日last；填充源不可得时退化为ffill；本轮未使用这些日内字段"},
    {"源字段": "acc_open / acc_close / close / 市值 / 普通因子字段", "规则": "ffill", "处理": "仅历史前填，包括跨显式NaN；首个非空之前不补；acc_open延续自身"},
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |"] +
        ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)])


def test_log_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    matches = list(re.finditer(r"Ran (\d+) tests? in ([\d.]+)s", text))
    final = matches[-1] if matches else None
    tail = text[final.end():].strip() if final else ""
    return {"path": str(path), "exists": path.is_file(),
        "tests": int(final.group(1)) if final else None,
        "seconds": float(final.group(2)) if final else None,
        "passed": bool(final and re.fullmatch(r"OK(?:\s*\([^\n]*\))?", tail)),
        "parsed_runs": len(matches), "fingerprint": fingerprint(path) if path.is_file() else None}


def _fingerprints(value):
    if isinstance(value, dict):
        if {"path", "size", "mtime_ns", "sampled_sha256"}.issubset(value):
            yield value
        for child in value.values():
            yield from _fingerprints(child)
    elif isinstance(value, list):
        for child in value:
            yield from _fingerprints(child)


def source_integrity(inputs: dict) -> list[dict]:
    """Verify recorded AIQ/Mongo source fingerprints without loading tables."""
    source_roots = (Path(inputs.get("aiq_root", "/data/AIQ")).resolve(), Path("/data/mongo").resolve())
    expected = {}
    for item in _fingerprints(inputs):
        path = Path(item["path"]).resolve()
        if any(path.is_relative_to(root) for root in source_roots):
            expected.setdefault(str(path), []).append(item)
    protected = inputs["price_source_protected"]
    expected.setdefault(str(Path(protected["path"]).resolve()), []).append(protected)
    rows = []
    for path_text, versions in sorted(expected.items()):
        path = Path(path_text)
        current = fingerprint(path) if path.is_file() else None
        fields = ("size", "mtime_ns", "sampled_sha256")
        distinct = {tuple(item[field] for field in fields) for item in versions}
        matches = current is not None and all(all(current[field] == item[field] for field in fields) for item in versions)
        rows.append({"path": path_text, "unchanged": matches, "recorded_versions": len(distinct),
            "size": current["size"] if current else None, "sampled_sha256": current["sampled_sha256"] if current else None,
            "status": "unchanged" if matches else "missing" if current is None else "fingerprint_mismatch",
            "method": "first/last 1 MiB + size + mtime_ns; not a full-file hash"})
    return rows


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        raise ValueError(f"Required new-profile coverage field missing: {column}")
    values = pd.to_numeric(frame[column], errors="raise")
    if values.isna().any():
        raise ValueError(f"Missing count in coverage field: {column}")
    return int(values.sum())


def _quality_value(quality: pd.DataFrame, metric: str):
    rows = quality.loc[quality["指标"].eq(metric), "记录数"]
    return int(rows.iloc[0]) if len(rows) == 1 else None


def read_run(run: Path, protected_price: Path, *, expect_matrix: bool = True) -> dict:
    result = read_json(run / "results.json")
    if result.get("status") != "completed":
        raise ValueError(f"Run is not completed: {run}")
    config = read_json(run / "config.frozen.json")
    method = config.get("missing_values", {}).get("method", "preserve")
    if expect_matrix and method != "aiq_matrix":
        raise ValueError(f"Not an aiq_matrix run: {run}")
    if Path(config["sources"]["quotes"]).resolve() != protected_price.resolve():
        raise ValueError(f"Unexpected return-price source: {run}")
    panel = read_json(run / "panel_manifest.json")
    frequency = config["backtest"]["rebalance_frequency"]
    periods = int(panel["periods"])
    coverage = pd.read_csv(run / "coverage.csv")
    quality = pd.read_csv(run / "data_quality_summary.csv")
    if coverage.duplicated(["factor", "entry"]).any():
        raise ValueError(f"Duplicate factor/entry coverage in {run}")
    expected_factors = [item["name"] for item in config["factors"]]
    if {item["name"] for item in result["factors"]} != set(expected_factors):
        raise ValueError(f"Missing configured factor result: {run}")
    fills, strategies = [], []
    for factor in result["factors"]:
        name = factor["name"]
        covered = coverage.loc[coverage.factor.eq(name)]
        if len(covered) != periods:
            raise ValueError(f"Incomplete coverage periods: {run}/{name}")
        if expect_matrix:
            fills.append({"frequency": frequency, "factor": name, "periods": periods,
                "base_stock_periods": _sum_column(covered, "base_count"),
                "factor_original_missing": _sum_column(covered, "factor_original_missing"),
                "factor_filled": _sum_column(covered, "factor_filled"),
                "factor_unavailable_after_fill": _sum_column(covered, "missing_factor"),
                "industry_filled": _sum_column(covered, "industry_filled"),
                "market_cap_filled": _sum_column(covered, "market_cap_filled")})
        ic = pd.read_csv(run / factor["output_dir"] / "ic_summary.csv")
        perf = pd.read_csv(run / factor["output_dir"] / "performance.csv")
        variants = factor.get("variants", list(factor.get("summary", {})))
        if expect_matrix and set(variants) != set(VARIANTS):
            raise ValueError(f"Expected the four configured variants: {run}/{name}")
        for variant in variants:
            rows = ic.loc[ic.variant.eq(variant) & ic.method.eq("spearman") & ic.subset.eq("ALL")]
            long = perf.loc[perf.variant.eq(variant) & perf.portfolio.eq("Q10")]
            spread = perf.loc[perf.variant.eq(variant) & perf.portfolio.eq("Q10-Q1")]
            if len(rows) != 1 or len(long) != 1 or len(spread) != 1:
                raise ValueError(f"Nonunique/missing IC or portfolio summary: {run}/{name}/{variant}")
            row, long_row, spread_row = rows.iloc[0], long.iloc[0], spread.iloc[0]
            strategies.append({"frequency": frequency, "factor": name, "variant": variant,
                "method": method, "rank_ic": row.ic_mean, "annual_icir": row.annual_icir,
                "rank_ic_periods": int(row.n), "rank_ic_nw_lower": row.get("nw_lower"),
                "rank_ic_nw_upper": row.get("nw_upper"),
                "long_excess_sharpe": long_row.get("excess_sharpe"),
                "long_excess_sharpe_periods": long_row.get("excess_sharpe_n_periods"),
                "long_short_excess_sharpe": spread_row.get("excess_sharpe"),
                "long_short_excess_sharpe_periods": spread_row.get("excess_sharpe_n_periods"),
                "long_mean_turnover_one_way": long_row.get("mean_turnover_one_way"),
                "long_p95_turnover_one_way": long_row.get("p95_turnover_one_way"),
                "long_high_turnover_fraction": long_row.get("high_turnover_fraction"),
                "long_turnover_observations": long_row.get("turnover_observations"),
                "long_turnover_missing_periods": long_row.get("turnover_missing_periods"),
                "turnover_warning_threshold": long_row.get("turnover_warning_threshold"),
                "signal_stock_periods": factor.get("sample", {}).get("variant_signal_rows", {}).get(variant),
                "report_path": str(run / factor["output_dir"] / "report.html")})
    summary = {"frequency": frequency, "missing_values": method, "periods": periods,
        "holding_start": str(panel["holding_start"])[:10], "holding_end": str(panel["holding_end"])[:10],
        "base_stock_periods": int(panel["base_rows"]),
        "unknown_st_base_stock_periods": int(panel["unknown_st_base_rows"]),
        "industry_available_after_fill": _quality_value(quality, "行业有效"),
        "industry_filled": _quality_value(quality, "行业AIQ填充"),
        "market_cap_available_after_fill": _quality_value(quality, "流通市值有效"),
        "market_cap_filled": _quality_value(quality, "流通市值AIQ填充"),
        "unresolved_labels": int(panel["unresolved_labels"]),
        "unresolved_delisting": _quality_value(quality, "退市退出未解决"),
        "entry_open_filled": _quality_value(quality, "entry开盘AIQ填充"),
        "exit_open_filled": _quality_value(quality, "exit开盘AIQ填充"),
        "factor_count": len(result["factors"]), "run_dir": str(run),
        "config_digest": result.get("config_digest"), "sample_policy": panel.get("sample_policy"),
        "smoothing": config.get("smoothing", {"method": "none"}), "preprocess": config.get("preprocess", {})}
    return {"summary": summary, "fills": fills, "strategies": strategies, "quality": quality,
        "data_usage": pd.read_csv(run / "data_usage.csv"),
        "input_fingerprints": {name: fingerprint(run / name) for name in
            ("results.json", "config.frozen.json", "panel_manifest.json", "coverage.csv", "data_quality_summary.csv")}}


def generate(prepared: Path, runs: list[Path], output: Path, test_log: Path,
             baselines: list[Path] | None = None) -> dict:
    prepared, output, test_log = prepared.resolve(), output.resolve(), test_log.resolve()
    inputs = read_json(prepared / "manifest.json")
    if inputs.get("missing_values", {}).get("method") != "aiq_matrix":
        raise ValueError("Prepared manifest is not an aiq_matrix manifest")
    protected = Path(inputs["price_source_protected"]["path"])
    immutable_roots = [Path(inputs.get("aiq_root", "/data/AIQ")).resolve(), Path("/data/mongo").resolve()]
    for protected_root in immutable_roots + [prepared] + [path.resolve() for path in runs + (baselines or [])]:
        if output.is_relative_to(protected_root) or protected_root.is_relative_to(output):
            raise ValueError("Report directory must be separate from source and run directories")
    # Read/check completed small result files before creating output artifacts.
    loaded = [read_run(path.resolve(), protected) for path in runs]
    if len({run["summary"]["frequency"] for run in loaded}) != len(loaded):
        raise ValueError("Supply only one new run per rebalance frequency")
    integrity = source_integrity(inputs)
    tests = test_log_summary(test_log)
    status_manifest = inputs["datasets"]["status"]
    status_validation = read_json(prepared / "status_validation.json")
    status_counts = status_manifest["counts"]
    status_rows = [{"field": name,
        "original_missing_rows": status_counts[f"{name}_original_missing_rows"],
        "filled_rows": status_counts[f"{name}_filled_rows"],
        "rule": status_manifest["field_rules"][name]} for name in ("open", "close", "volume", "last")]
    counts_match = (status_validation.get("rows_verified") == status_counts["rows"] and
        all(status_validation.get("filled_counts", {}).get(f"{row['field']}_filled_rows") == row["filled_rows"]
            for row in status_rows))
    status_passed = (status_validation.get("status") == "passed" and counts_match and
        not any(status_validation.get("violations", {}).values()))
    validation_artifact = status_validation.get("artifact", {})
    status_artifact = status_manifest.get("output", {})
    status_artifact_match = all(validation_artifact.get(field) == status_artifact.get(field)
                                for field in ("size", "mtime_ns", "sampled_sha256"))
    status_passed &= status_artifact_match
    old = []
    baseline_notes = []
    for path in baselines or []:
        if not (path / "results.json").is_file():
            baseline_notes.append(f"未提供可读取的历史完成结果：{path}")
            continue
        previous = read_run(path.resolve(), protected, expect_matrix=False)
        if previous["summary"]["missing_values"] != "preserve":
            baseline_notes.append(f"跳过非preserve历史口径：{path}")
            continue
        old.append(previous)
    run_rows = [run["summary"] for run in loaded]
    fills = [row for run in loaded for row in run["fills"]]
    strategies = [row for run in loaded for row in run["strategies"]]
    old_lookup = {(row["frequency"], row["factor"], row["variant"]): row
                  for run in old for row in run["strategies"]}
    comparison = []
    for row in strategies:
        baseline = old_lookup.get((row["frequency"], row["factor"], row["variant"]))
        if baseline is not None:
            comparison.append({key: row[key] for key in ("frequency", "factor", "variant")} | {
                "old_method": "preserve", "new_method": "aiq_matrix",
                "old_signal_stock_periods": baseline["signal_stock_periods"],
                "new_signal_stock_periods": row["signal_stock_periods"],
                "old_rank_ic": baseline["rank_ic"], "new_rank_ic": row["rank_ic"],
                "old_long_excess_sharpe": baseline["long_excess_sharpe"],
                "new_long_excess_sharpe": row["long_excess_sharpe"],
                "old_long_short_excess_sharpe": baseline["long_short_excess_sharpe"],
                "new_long_short_excess_sharpe": row["long_short_excess_sharpe"]})
    validation_passed = bool(integrity and all(row["unchanged"] for row in integrity) and tests["passed"] and status_passed)
    summary = {"schema_version": 1, "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile": "aiq_matrix", "validation_passed": validation_passed,
        "inputs_manifest": fingerprint(prepared / "manifest.json"), "source_integrity": integrity,
        "tests": tests, "status_validation_passed": bool(status_passed),
        "status_validation_artifact_matches_manifest": status_artifact_match,
        "status_validation": status_validation, "status_counts": status_counts,
        "status_fill_summary": status_rows, "runs": run_rows, "factor_fill_summary": fills,
        "strategies": strategies, "baseline_comparison": comparison, "baseline_notes": baseline_notes,
        "run_input_fingerprints": {row["summary"]["frequency"]: row["input_fingerprints"] for row in loaded},
        "limitations": ["filled is an imputation flag, not independently observed or certified known data",
            "Generated zero volume is a suspension proxy assumption, not evidence of an actual full-day halt",
            "Historical preserve/new aiq_matrix comparisons change methods, inputs and sometimes samples; they are not causal effects",
            "Source checks use sampled fingerprints and metadata; not full-file content hashes",
            "Only small summaries are read; per-row panel acceptance is not re-run by this report generator"]}
    sections, markdown = [], []
    def heading(text):
        sections.append(f"<h2>{html.escape(text)}</h2>")
        markdown.append("## " + text)
    def paragraph(text):
        sections.append(f"<p>{html.escape(text)}</p>")
        markdown.append(text)
    def table(rows, rename=None):
        frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if frame.empty:
            paragraph("无对应记录。")
            return
        if rename:
            frame = frame.rename(columns=rename)
        sections.append('<div class="scroll">' + frame.to_html(index=False, escape=True, border=0,
            na_rep="—", float_format=lambda value: f"{value:.6g}") + "</div>")
        markdown.append(md_table(frame))
    def link(label, path):
        relative = Path(os.path.relpath(path, output)).as_posix()
        sections.append(f'<p><a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a></p>')
        markdown.append(f"[{label}]({relative})")
    heading("AIQ矩阵补缺核验与因子测试")
    paragraph(f"新口径：missing_values.method=aiq_matrix。生成时间：{summary['generated_utc']}。"
              "本报告只汇总已完成运行和小型审计文件，不重新读取逐股持仓/特征大表，不重新计算因子或收益。")
    paragraph("本轮验证状态：" + ("通过。" if validation_passed else "存在未通过或缺失的验证项，详见后文；不能据此称验收完成。"))
    heading("字段规则与旧preserve口径的区别")
    table(RULE_ROWS)
    paragraph("aiq_matrix允许按字段规则跨显式NaN填充，保留原值、填充标记和实际供值日；preserve保留最新记录的显式缺失。"
              "省略missing_values仍为preserve，旧冻结配置和报告不被改写。零、负值与无穷值不是可跳过的NaN，经济有效性检查另行执行。"
              "没有叠加getfactor的行业×总市值回归补缺；EMA、缩尾、标准化和中性化仍由各自配置控制。")
    paragraph("行业按35类规则前填的是测试类别：filled不能解释为已获得真实新分类，原始区间缺失与冲突仍保留。"
              "ST按实施/取消与证据到期事件处理，不通用前填未知。源行情文件保持只读，收益使用已修复复权价格。"
              "退市后未观察的旧价不能代替退出现金；未来周/月收益不一律补0。")
    heading("状态输入的原始缺失与填充")
    paragraph(f"状态矩阵共 {status_counts['rows']:,} 个股票日、{status_manifest['codes']:,} 只股票；"
              f"其中 {status_counts['no_raw_quote_key_rows']:,} 个股票日没有原行情键。以下为矩阵层统计，不是调仓池股周/月数。")
    table(status_rows, {"field": "字段", "original_missing_rows": "原始缺失记录", "filled_rows": "填充记录", "rule": "AIQ规则"})
    paragraph("volume补0后按GF量价规则产生的停牌标记仍是代理假设，不代表查询到了真实全天停牌记录；"
              "raw open保持缺失，close/last前填也不能被称为实际成交。矩阵验收检查未来供值、上市前供值、生命周期及计数一致性。")
    table([{"状态矩阵验证": status_validation.get("status"), "核验行数": status_validation.get("rows_verified"),
        "违规项目合计": sum(status_validation.get("violations", {}).values()), "清单及产物一致": bool(counts_match and status_artifact_match)}])
    link("状态矩阵逐项核验JSON", prepared / "status_validation.json")
    heading("完成运行与基础样本")
    visible_run = [{key: row[key] for key in ("frequency", "periods", "holding_start", "holding_end", "base_stock_periods",
                    "unknown_st_base_stock_periods", "unresolved_labels", "unresolved_delisting")} for row in run_rows]
    table(visible_run, {"frequency": "调仓", "periods": "完整期数", "holding_start": "入场起", "holding_end": "最后退出",
        "base_stock_periods": "基础股期次", "unknown_st_base_stock_periods": "基础池ST未知", "unresolved_labels": "收益未解决", "unresolved_delisting": "退市退出未解决"})
    paragraph("期数取自各次完成运行，不预设为697或162。基础样本和ST未知数来自面板清单；下列质量表的每个指标使用其各自分母，不能混用。")
    for run in loaded:
        frequency = run["summary"]["frequency"]
        heading(FREQUENCY_NAMES.get(frequency, frequency) + "的数据质量与填充覆盖")
        table(run["quality"])
        paragraph("行业有效/市值有效表示填充及检查后的可用性，不等于原始观察完整；AIQ填充行已另列。")
        link(FREQUENCY_NAMES.get(frequency, frequency) + "完整因子测试报告", Path(run["summary"]["run_dir"]) / "report.html")
    heading("两个因子的原始缺失、填充与剩余缺失")
    table(fills, {"frequency": "调仓", "factor": "因子", "periods": "期数", "base_stock_periods": "基础股期次",
        "factor_original_missing": "原因子缺失", "factor_filled": "因子已填充", "factor_unavailable_after_fill": "填后无效因子",
        "industry_filled": "行业已填充", "market_cap_filled": "市值已填充"})
    paragraph("这些计数对coverage.csv按因子分别汇总。行业/市值计数在两个因子行中重复使用同一基础池，不能跨因子相加；"
              "原始缺失、填后无效和非法非空值属于不同定义，也不能不经核对直接做减法。")
    heading("各版本RankIC、换手与超额夏普")
    metric_columns = ["frequency", "factor", "variant", "rank_ic", "rank_ic_periods", "annual_icir", "long_excess_sharpe",
        "long_short_excess_sharpe", "long_mean_turnover_one_way", "long_p95_turnover_one_way", "long_high_turnover_fraction",
        "long_turnover_observations", "long_turnover_missing_periods"]
    table(pd.DataFrame(strategies)[metric_columns], {"frequency": "调仓", "factor": "因子", "variant": "版本",
        "rank_ic": "平均RankIC", "rank_ic_periods": "IC有效期数", "annual_icir": "年化ICIR",
        "long_excess_sharpe": "多头超额夏普", "long_short_excess_sharpe": "多空超额夏普",
        "long_mean_turnover_one_way": "多头平均单边换手", "long_p95_turnover_one_way": "多头单边换手P95",
        "long_high_turnover_fraction": "换手超警戒比例", "long_turnover_observations": "换手有效期数", "long_turnover_missing_periods": "换手缺失期数"})
    paragraph("raw/industry/size/industry_size分别为未中性化、行业、市值、行业市值版本。RankIC取全样本Spearman逐期均值。"
              "多头超额夏普用Q10−基准；多空为Q10−Q1，两腿基准抵消。单边换手及比例为小数，平均/P95不包括首次建仓，缺失不计作零。"
              "夏普采用算术期均值/样本标准差×年化频数平方根，独立样本版本差异不全来自中性化。")
    for row in strategies:
        if row["variant"] == "raw":
            link(f"{FREQUENCY_NAMES.get(row['frequency'], row['frequency'])} · {row['factor']} · 四版本详情", Path(row["report_path"]))
    if comparison:
        heading("与历史preserve结果对照")
        table(comparison)
        paragraph("这是不同缺失策略下既有研究结果的并列比较。矩阵规则同时影响因子、行业、市值、价格及停牌代理，基础池和各版本样本亦可能改变；"
                  "不是固定样本单变量实验，不能将收益/IC变化解释为补缺的因果效果或已证实改进。")
    for note in baseline_notes:
        paragraph(note)
    heading("源保护与测试验证")
    table(pd.DataFrame(integrity)[["path", "unchanged", "recorded_versions", "status"]],
          {"path": "原始来源", "unchanged": "指纹未变", "recorded_versions": "清单版本数", "status": "核验状态"})
    paragraph("源保护重新核对清单中的AIQ/Mongo原始来源及已修复行情。仅读取首尾各1MiB并比较大小、mtime与抽样SHA256，"
              "不将其称为整个大文件逐字节证明；适配产物与程序文件不混入原始源清单。")
    table([{"测试日志": str(test_log), "最后一次测试数": tests["tests"], "耗时秒": tests["seconds"], "结果OK": tests["passed"]}])
    paragraph("测试数量和OK状态直接解析日志的最后一次unittest结果，不硬编码311，也不把重复运行相加。"
              "本汇总生成器没有再扫描巨型持仓或特征表；已完成的逐行状态检查由status_validation.json提供。")
    link("实际测试日志", test_log)
    heading("保留的数据限制")
    for note in inputs.get("remaining_improvements", []):
        paragraph(note)
    paragraph("新报告与旧结果分别保存，所有来源、字段和缺失处理以各自冻结配置为准。详细机器可读数据见summary.json；"
              "run_summary.csv、factor_fill_summary.csv、strategy_summary.csv可直接用于后续验证说明。")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("run_summary.csv", run_rows), ("factor_fill_summary.csv", fills), ("strategy_summary.csv", strategies),
                       ("status_fill_summary.csv", status_rows), ("source_integrity.csv", integrity), ("baseline_comparison.csv", comparison)):
        pd.DataFrame(rows).to_csv(output / name, index=False)
    write_json(output / "summary.json", summary)
    (output / "AIQ矩阵补缺核验报告.md").write_text("\n\n".join(markdown) + "\n", encoding="utf-8")
    style = "body{font:15px/1.75 system-ui,sans-serif;color:#253545;background:#f3f6f9;margin:0}main{max-width:1500px;margin:24px auto;padding:32px;background:white}h2{color:#164f72;border-bottom:1px solid #d9e4ed;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border:1px solid #dbe3e9;text-align:left}th{background:#edf4fa}.scroll{overflow:auto}a{color:#176592}p{max-width:1300px}@media print{main{margin:0;padding:0}.scroll{overflow:visible}}"
    (output / "report.html").write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>AIQ矩阵补缺核验</title><style>' + style +
        "</style></head><body><main>" + "\n".join(sections) + "</main></body></html>", encoding="utf-8")
    print(output / "report.html")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PROJECT / "outputs/aiq_matrix_inputs_20261028")
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/aiq_matrix_validation_20261028")
    parser.add_argument("--test-log", type=Path, default=PROJECT / "outputs/aiq_matrix_tests_final_20261028.log")
    parser.add_argument("--baseline", type=Path, action="append", help="Optional completed preserve run; repeat for each frequency")
    parser.add_argument("--with-baselines", action="store_true", help="Compare existing standard weekly/monthly preserve runs")
    args = parser.parse_args()
    run_paths = args.run_dir or [PROJECT / "outputs/factor_test_weekly_aiq_matrix_20261028",
                                PROJECT / "outputs/factor_test_monthly_aiq_matrix_20261028"]
    baseline_paths = args.baseline or []
    if args.with_baselines:
        baseline_paths += [PROJECT / "outputs/factor_test_aiq_weekly", PROJECT / "outputs/factor_test_aiq_monthly"]
    result = generate(args.prepared, run_paths, args.output, args.test_log, baseline_paths)
    raise SystemExit(0 if result["validation_passed"] else 2)
