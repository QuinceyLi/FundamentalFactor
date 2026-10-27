"""Offline, self-contained reports for the schema-v2 single-factor backtest."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_BACKTEST, DEFAULT_METRICS, DEFAULT_PREPROCESS, RISK_FIELDS


VARIANTS = {"raw": "原始信号", "industry_size": "行业市值中性化信号", "full": "全中性化信号"}


def _variant_labels(config: dict | None = None, values=()) -> dict:
    all_labels = VARIANTS | {"industry": "行业中性化信号"}
    preprocessing = (config or {}).get("preprocess", {})
    if not preprocessing.get("winsorize", True) and not preprocessing.get("standardize", True):
        all_labels["raw"] = "原样输入信号"
    industry_only = (config or {}).get("neutralization", {}).get("include_float_mcap", True) is False
    if industry_only or "industry" in values:
        labels = {name: all_labels[name] for name in ("raw", "industry", "full")}
    else:
        labels = {name: all_labels[name] for name in VARIANTS}
    selected = (config or {}).get("neutralization", {}).get("variants")
    if selected is not None:
        return {name: all_labels[name] for name in selected if name in all_labels}
    if len(values):
        return {name: all_labels[name] for name in values if name in all_labels}
    return labels


def _independent_samples(config: dict) -> bool:
    return config.get("neutralization", {}).get("sample_policy", "common") == "independent"


def _contextualize(body: str, config: dict) -> str:
    """Adapt generated report prose, leaving source audit and filenames intact."""
    if config.get("backtest", {}).get("rebalance_frequency", "weekly") == "monthly":
        for old, new in (("下一周", "下一月"), ("周度", "月度"), ("每周", "每月"),
                         ("逐周", "逐月"), ("当周", "当月"), ("整周", "整月"),
                         ("周初", "月初"), ("周内", "月内"), ("股周", "股月"),
                         ("周收益", "月收益"), ("周数", "月数"), ("周次", "月次"),
                         ("该周", "该月"), ("各周", "各月"), ("单周", "单月"),
                         ("受影响周起", "受影响月起"), (" 周）", " 月）")):
            body = body.replace(old, new)
    if _independent_samples(config):
        for old, new in (("所选信号共同样本数", "控制字段交集数（诊断）"),
                         ("共同样本覆盖率", "控制字段交集覆盖率（诊断）"),
                         ("共同样本 ALL", "该版本样本 ALL"),
                         ("ALL（共同样本多头）", "ALL（该版本样本多头）"),
                         ("该因子共同样本的独立多头账户", "该版本样本的独立多头账户"),
                         ("ALL 是共同样本的独立多头账户", "ALL 是该版本样本的独立多头账户")):
            body = body.replace(old, new)
    return body


def _coverage_series(frame: pd.DataFrame, config: dict, prefix: str = "") -> list[dict]:
    ordered = frame.sort_values("entry") if "entry" in frame else frame
    series = [{"name": prefix + "原始因子覆盖率", "points": _points(ordered, "raw_coverage", "entry")}]
    if _independent_samples(config):
        for name, label in _variant_labels(config).items():
            column = f"variant_{name}_coverage"
            if column in ordered:
                series.append({"name": prefix + label + "测试覆盖率", "points": _points(ordered, column, "entry")})
        if len(_variant_labels(config)) > 1 and "common_coverage" in ordered:
            series.append({"name": prefix + "控制字段交集（仅诊断）", "points": _points(ordered, "common_coverage", "entry")})
    else:
        series.append({"name": prefix + "所选信号共同样本覆盖率", "points": _points(ordered, "common_coverage", "entry")})
    return series


def _universe_label(config: dict) -> str:
    exchanges = config.get("universe", {}).get("exchanges")
    if exchanges is None or set(exchanges) == {"SH", "SZ", "BJ"}:
        return "历史全 A"
    return "".join({"SH": "沪", "SZ": "深", "BJ": "北"}[x] for x in exchanges) + " A 股"
SUBSETS = ["ALL", "TOP20", "TOP50", "BOTTOM50", "BOTTOM20"]
PORTFOLIOS = [f"Q{i}" for i in range(1, 11)] + ["Q10-Q1"]
COLORS = ["#155e9b", "#028090", "#48a9a6", "#8caa45", "#beaa37", "#dc8a32", "#db6255", "#a94365", "#7455a0", "#282553"]
LABELS = {
    "variant": "信号", "method": "相关系数", "subset": "样本", "portfolio": "组合", "year": "年份",
    "n": "总期数", "n_periods": "总期数", "ic_mean": "IC均值", "ic_std": "IC波动率",
    "annual_icir": "年化ICIR", "t_stat": "普通t值", "win_rate": "胜率", "nw_t": "NW t值",
    "nw_lower": "NW 95%下限", "nw_upper": "NW 95%上限", "nw_status": "NW状态",
    "total_return": "总收益", "annual_return": "年化收益", "annual_volatility": "年化波动率",
    "information_ratio": "信息比率", "max_drawdown": "最大回撤", "annual_turnover": "年化换手率（双边）",
    "annual_target_turnover": "年化目标换手率（双边）",
    "excess_total_return": "超额总收益", "excess_annual_return": "超额年化收益",
    "excess_annual_volatility": "超额年化波动率", "excess_information_ratio": "超额信息比率",
    "excess_win_rate": "超额胜率", "excess_max_drawdown": "超额最大回撤", "status": "状态",
    "entry": "入场日", "exit": "退出日", "factor": "因子", "base_count": "基准股票数",
    "factor_available": "因子有效股票数", "common_count": "所选信号共同样本数", "raw_coverage": "原始覆盖率",
    "common_coverage": "共同样本覆盖率", "missing_factor": "缺失因子数", "missing_controls": "缺失控制数",
    "zero_fraction": "零值比例", "tie_fraction": "并列值比例", "n_stocks": "股票数", "reason": "原因",
    "calendar_weeks": "日历周数", "valid_return_weeks": "有效收益周数", "empty_weeks": "空组/空腿周数",
    "missing_return_weeks": "缺失收益周数", "mean_stocks": "平均股票数", "min_stocks": "最少股票数",
    "max_stocks": "最多股票数", "mean_restricted_fraction": "平均交易受限比例",
    "max_restricted_fraction": "最高交易受限比例", "rank": "控制矩阵秩", "columns": "控制列数",
    "redundant_columns": "冗余控制列", "winsor_method": "截尾方法",
    "mean_cash_weight": "平均周初现金权重", "max_cash_weight": "最高周初现金权重",
    "target_empty_weeks": "空目标周数", "signal_missing_weeks": "缺失信号冻结周数",
    "blocked_buy_weeks": "买入未完成周数", "blocked_sell_weeks": "卖出未完成周数",
    "blocked_buy_value": "累计未成交买入金额", "blocked_sell_value": "累计未成交卖出金额",
    "carried_valuation_count": "累计沿用估值次数", "rebalance_weeks": "持有周数",
    "n_labels": "有效收益标签数", "label_coverage": "收益标签覆盖率",
    "unknown_st_count": "ST 未知股票数", "unknown_st_fraction": "ST 未知比例",
    "trade_restriction_unknown_fraction": "交易限制未知比例",
    "incomplete_label_weeks": "含缺失标签周数", "mean_missing_weight": "平均缺失标签目标权重",
    "max_missing_weight": "最高缺失标签目标权重", "mean_evaluation_weight": "平均统计权重合计",
    "data_complete": "标签完整", "missing_weight": "缺失标签目标权重",
    "evaluation_weight_sum": "统计权重合计", "cash_weight": "现金目标权重",
    "return_method": "收益统计方法", "target_turnover": "目标双边换手率",
}
PERCENT = {
    "ic_mean", "ic_std", "win_rate", "nw_lower", "nw_upper", "total_return", "annual_return",
    "annual_volatility", "max_drawdown", "annual_turnover", "excess_total_return", "excess_annual_return",
    "excess_annual_volatility", "excess_win_rate", "excess_max_drawdown", "raw_coverage", "common_coverage",
    "zero_fraction", "tie_fraction", "finite_fraction", "missing_fraction",
    "mean_restricted_fraction", "max_restricted_fraction",
    "mean_cash_weight", "max_cash_weight",
    "label_coverage",
    "unknown_st_fraction", "trade_restriction_unknown_fraction",
    "annual_target_turnover", "target_turnover", "mean_missing_weight", "max_missing_weight",
    "mean_evaluation_weight", "missing_weight", "evaluation_weight_sum", "cash_weight",
}
for _variant, _label in (VARIANTS | {"industry": "行业中性化信号"}).items():
    LABELS[f"variant_{_variant}_count"] = f"{_label}测试股票数"
    LABELS[f"variant_{_variant}_coverage"] = f"{_label}测试覆盖率"
    PERCENT.add(f"variant_{_variant}_coverage")
PERFORMANCE_COLUMNS = ["portfolio", "n_periods", "total_return", "annual_return", "annual_volatility",
                       "information_ratio", "win_rate", "max_drawdown", "annual_turnover", "excess_total_return",
                       "excess_annual_return", "excess_annual_volatility", "excess_information_ratio",
                       "excess_win_rate", "excess_max_drawdown", "status"]
IC_COLUMNS = ["subset", "n", "ic_mean", "ic_std", "annual_icir", "t_stat", "win_rate", "nw_t", "nw_lower", "nw_upper", "nw_status"]


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, ValueError):
        return pd.DataFrame()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value: Any) -> float | None:
    return float(value) if _finite(value) else None


def _format(value: Any, column: str = "") -> str:
    if value is None or (not isinstance(value, (dict, list, tuple)) and pd.isna(value)):
        return "—"
    if isinstance(value, (dict, list, tuple)):
        return _escape(json.dumps(value, ensure_ascii=False, default=str))
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(value, bool):
        if not _finite(value):
            return "—"
        if column in PERCENT:
            return f"{float(value):.2%}"
        if column.startswith("variant_") and column.endswith("_count") or column in {"n", "n_periods", "year", "n_stocks", "base_count", "common_count", "factor_available", "missing_factor", "missing_controls",
                      "calendar_weeks", "valid_return_weeks", "empty_weeks", "missing_return_weeks", "min_stocks", "max_stocks",
                      "target_empty_weeks", "signal_missing_weeks", "blocked_buy_weeks", "blocked_sell_weeks",
                      "carried_valuation_count", "rebalance_weeks"}:
            return f"{int(value):,}"
        return f"{float(value):.4g}"
    return _escape(value)


def render_table(frame: pd.DataFrame, limit: int = 100, columns: list[str] | None = None) -> str:
    if frame.empty:
        return '<p class="muted">暂无记录。</p>'
    selected = [c for c in (columns or list(frame.columns)) if c in frame]
    body = []
    for values in frame.head(limit)[selected].itertuples(index=False, name=None):
        body.append("<tr>" + "".join(f"<td>{_format(value, column)}</td>" for value, column in zip(values, selected)) + "</tr>")
    head = "".join(f"<th>{_escape(LABELS.get(column, column))}</th>" for column in selected)
    tail = f'<p class="muted">展示前 {limit} 条，完整数据见 CSV。</p>' if len(frame) > limit else ""
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{tail}'


def _details(title: str, body: str, opened: bool = False) -> str:
    return f'<details{" open" if opened else ""}><summary>{_escape(title)}</summary>{body}</details>'


def _notice(title: str, text: str, kind: str = "warning") -> str:
    return f'<aside class="notice {kind}"><strong>{_escape(title)}</strong><p>{_escape(text)}</p></aside>'


def _date(value: Any) -> str:
    try:
        parsed = pd.Timestamp(value)
        return parsed.strftime("%Y-%m-%d") if not pd.isna(parsed) else ""
    except (TypeError, ValueError):
        return ""


class _Charts:
    """Small inline SVG charts; all hover data stays in the HTML document."""

    def __init__(self) -> None:
        self.count = 0

    def chart(self, title: str, series: list[dict], *, kind: str = "line", unit: str = "number", caption: str = "") -> str:
        self.count += 1
        identifier = f"chart-{self.count}"
        dates = sorted({p["date"] for s in series for p in s["points"] if p.get("date")})
        finite = [float(p["value"]) for s in series for p in s["points"] if _finite(p.get("value"))]
        if not dates or not finite:
            return f'<section class="chart"><h3>{_escape(title)}</h3><p class="muted">没有可绘制的有效值；缺失值未以零替代。</p></section>'
        categories = kind == "category"
        if categories:
            dates = list(dict.fromkeys(p["date"] for s in series for p in s["points"]))
            positions = {day: float(i) for i, day in enumerate(dates)}
        else:
            positions = {day: float(pd.Timestamp(day).toordinal()) for day in dates}
        xmin, xmax = min(positions.values()), max(positions.values())
        low, high = min(finite), max(finite)
        if kind in {"bar", "category"}:
            low, high = min(low, 0), max(high, 0)
        padding = max((high - low) * .08, abs(high) * .02, .001)
        low, high = low - padding, high + padding
        left, right, top, bottom = 74., 940., 18., 265.

        def x(day: str) -> float:
            if categories:
                return left + (right - left) * (positions[day] + .5) / max(len(dates), 1)
            return left + (right - left) * (positions[day] - xmin) / max(xmax - xmin, 1.)

        def y(value: float) -> float:
            return bottom - (bottom - top) * (value - low) / (high - low)

        svg = []
        for index in range(5):
            tick = low + (high - low) * index / 4
            py = y(tick)
            label = f"{tick:.1%}" if unit == "percent" else f"{tick:.3g}"
            svg.append(f'<line x1="{left}" x2="{right}" y1="{py:.2f}" y2="{py:.2f}" class="grid"/><text x="{left - 8}" y="{py + 4:.2f}" text-anchor="end">{label}</text>')
        tick_dates = dates if categories else [dates[i] for i in sorted({round(i * (len(dates) - 1) / min(4, max(len(dates) - 1, 1))) for i in range(min(5, len(dates)))})]
        for day in tick_dates:
            svg.append(f'<text x="{x(day):.2f}" y="291" text-anchor="middle">{_escape(day)}</text>')
        zero = y(0)
        if low <= 0 <= high:
            svg.append(f'<line x1="{left}" x2="{right}" y1="{zero:.2f}" y2="{zero:.2f}" class="zero"/>')
        payload_series = []
        legends = []
        for si, item in enumerate(series):
            color = COLORS[si % len(COLORS)]
            label = str(item["name"])
            legends.append(f'<button type="button" class="legend" data-series="{si}" aria-pressed="true" style="--series-color:{color}">{_escape(label)}</button>')
            points = item["points"]
            encoded = []
            if kind in {"bar", "category"}:
                width = min(45., (right - left) / max(len(dates), 1) * .78 / max(len(series), 1))
                for point in points:
                    value = _number(point.get("value"))
                    if value is None or point["date"] not in positions:
                        continue
                    px, py = x(point["date"]), y(value)
                    offset = (si - (len(series) - 1) / 2) * width
                    svg.append(f'<rect data-series="{si}" x="{px + offset - width / 2:.2f}" y="{min(py, zero):.2f}" width="{max(width, .4):.2f}" height="{max(abs(py - zero), .35):.2f}" fill="{color}" opacity=".82"/>')
                    encoded.append({"d": point["date"], "v": value, "x": round(px, 3), "y": round(py, 3)})
            else:
                path: list[str] = []
                segment = None

                def flush() -> None:
                    if path:
                        dash = ' stroke-dasharray="6 3"' if segment not in (None, 0, "0") else ""
                        svg.append(f'<path data-series="{si}" d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="1.7"{dash}/>')
                        if len(path) == 1:
                            cx, cy = path[0][1:].split(",")
                            svg.append(f'<circle data-series="{si}" cx="{cx}" cy="{cy}" r="2.5" fill="{color}"/>')
                        path.clear()

                for point in points:
                    value = _number(point.get("value"))
                    if value is None or point.get("date") not in positions:
                        flush()
                        segment = None
                        continue
                    current = point.get("segment", 0)
                    if current != segment:
                        flush()
                        segment = current
                    px, py = x(point["date"]), y(value)
                    path.append(f"{'M' if not path else 'L'}{px:.2f},{py:.2f}")
                    if point.get("reset"):
                        svg.append(f'<rect data-series="{si}" x="{px - 3:.2f}" y="{py - 3:.2f}" width="6" height="6" fill="{color}"><title>{_escape(point["date"])} 区段起点，净值归一为 1</title></rect>')
                    encoded.append({"d": point["date"], "v": value, "x": round(px, 3), "y": round(py, 3), "reset": bool(point.get("reset"))})
                flush()
            payload_series.append({"name": label, "color": color, "points": encoded})
        payload = json.dumps({"series": payload_series, "unit": unit, "axis": [{"d": day, "x": round(x(day), 3)} for day in dates]}, ensure_ascii=False, allow_nan=False)
        payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        svg.append('<line class="crosshair" x1="0" x2="0" y1="18" y2="265" visibility="hidden"/>')
        return (f'<section class="chart" id="{identifier}"><h3>{_escape(title)}</h3><div class="legends">{"".join(legends)}</div>'
                f'<div class="plot"><svg viewBox="0 0 970 310" role="img" aria-label="{_escape(title)}">{"".join(svg)}</svg>'
                '<div class="tooltip" hidden></div></div>'
                f'<p class="caption">{_escape(caption)}</p><script type="application/json" class="chart-data">{payload}</script></section>')


def _points(frame: pd.DataFrame, column: str, date_column: str = "exit") -> list[dict]:
    if frame.empty or column not in frame or date_column not in frame:
        return []
    return [{"date": _date(date), "value": _number(value)} for date, value in zip(frame[date_column], frame[column])]


def _nav_points(frame: pd.DataFrame, column: str) -> list[dict]:
    """Insert a visible NAV=1 start for every uninterrupted holding segment."""
    points = []
    previous_segment, previous_exit = None, None
    rendered_segment = -1
    for row in frame.sort_values("entry").to_dict("records"):
        value = _number(row.get(column))
        entry, exit_date = _date(row.get("entry")), _date(row.get("exit"))
        segment_key = "excess_segment_id" if column == "excess_nav" and "excess_segment_id" in row else "segment_id"
        segment = row.get(segment_key, 0)
        complete = row.get("complete", True)
        complete = str(complete).lower() not in {"false", "0", "nan", "none"}
        if value is None or not complete or not entry or not exit_date:
            points.append({"date": exit_date, "value": None})
            previous_segment, previous_exit = None, None
            continue
        if previous_exit != entry or previous_segment != segment:
            if points:
                points.append({"date": entry, "value": None})
            rendered_segment += 1
            points.append({"date": entry, "value": 1., "segment": rendered_segment, "reset": True})
        points.append({"date": exit_date, "value": value, "segment": rendered_segment})
        previous_segment, previous_exit = segment, exit_date
    return points


def _records(value: Any) -> str:
    if isinstance(value, dict):
        return render_table(pd.DataFrame([{"项目": key, "值": item} for key, item in value.items()]), 100)
    if isinstance(value, list):
        return render_table(pd.DataFrame(value), 100)
    return f"<p>{_escape(value)}</p>" if value else ""


def _sample_records(sample: dict, config: dict) -> str:
    """Present period metadata without exposing duplicate legacy week counters."""
    sample = dict(sample)
    frequency = config.get("backtest", {}).get("rebalance_frequency", "weekly")
    values = {"调仓频率": "月度" if frequency == "monthly" else "周度"}
    periods = sample.get("periods", sample.get("weeks"))
    if periods is not None:
        values["持有期数"] = periods
    names = {"signal_rows": "因子股票期记录数", "signal_start": "首个信号截点",
             "signal_end": "最后信号截点", "holding_start": "首次入场日", "holding_end": "最后退出日",
             "mean_raw_coverage": "因子平均有效覆盖率",
             "mean_common_coverage": "控制字段交集平均覆盖率（诊断）" if _independent_samples(config) else "共同样本平均覆盖率"}
    for key, value in sample.items():
        if key in {"periods", "weeks", "rebalance_frequency", "frequency"}:
            continue
        values[names.get(key, key)] = _format(value, "raw_coverage") if key.endswith("coverage") else value
    return _records(values)


def _source_audit(audit: dict) -> str:
    sources, numeric = [], []
    entries = audit.get("sources", {})
    if not isinstance(entries, dict):
        entries = {}
    for name, info in entries.items():
        if not isinstance(info, dict):
            continue
        fp = info.get("fingerprint", {})
        sources.append({"数据源": name, "文件": info.get("path", fp.get("path") if isinstance(fp, dict) else None),
                        "行数": info.get("rows"), "股票数": info.get("codes"), "起始日": info.get("start"), "末日": info.get("end"),
                        "指纹": fp})
        for column, stats in info.get("numeric", {}).items():
            row = {"数据源": name, "字段": column, **stats}
            rows = info.get("rows", 0) or 0
            if rows:
                row["finite_fraction"] = stats.get("finite", 0) / rows
                row["zero_fraction"] = stats.get("zero", 0) / rows
            numeric.append(row)
    warnings = "".join(f"<li>{_escape(item)}</li>" for item in audit.get("warnings", []))
    return ("<h2>数据来源与审计</h2>" + render_table(pd.DataFrame(sources))
            + "<h3>字段覆盖、零值及数值范围</h3>" + render_table(pd.DataFrame(numeric))
            + f'<ul>{warnings}</ul><p class="muted">抽样指纹用于版本追溯，不等同于全文件内容哈希。完整审计与字段映射保存在机器可读产物中。</p>')


def _definitions(config: dict) -> str:
    backtest = DEFAULT_BACKTEST | config.get("backtest", {})
    preprocess = DEFAULT_PREPROCESS | config.get("preprocess", {})
    monthly = backtest.get("rebalance_frequency", "weekly") == "monthly"
    metric_defaults = DEFAULT_METRICS | ({"annualization": 12, "nw_lag": 3} if monthly else {})
    metrics = metric_defaults | config.get("metrics", {})
    minimum = backtest["min_listing_sessions"]
    annualization, lag = metrics["annualization"], metrics["nw_lag"]
    quantiles = preprocess["fallback_quantiles"]
    risk_fields = config.get("risk_fields", RISK_FIELDS)
    multiplier, normalizer = preprocess["mad_multiplier"], preprocess["mad_normalizer"]
    tolerance = preprocess["rank_tolerance"]
    universe = _universe_label(config)
    variants = _variant_labels(config)
    include_cap = ("industry_size" in variants or
                   ("full" in variants and config.get("neutralization", {}).get("include_float_mcap", True)))
    neutral_definition = ("行业市值中性使用行业和个股实际流通市值的自然对数" if include_cap else
                          "纯行业中性仅使用行业标签和截距，不需要行业市值或个股实际市值")
    if set(variants) == {"raw"}:
        neutral_definition = "本次只测试原始信号，不要求行业、市值或 Barra 风险暴露"
    elif "full" in variants:
        neutral_definition += f"；全中性再加入冻结配置的全部 {len(risk_fields)} 项 Barra 暴露"
    selected_label = "、".join(variants.values())
    preprocessing_steps = []
    if preprocess.get("winsorize", True):
        preprocessing_steps.append(f"{multiplier:g}×{normalizer:g} MAD 截尾；MAD 为零时回退到 {quantiles[0] * 100:g}%/{quantiles[1] * 100:g}% 分位截尾")
    else:
        preprocessing_steps.append("去极值关闭")
    if preprocess.get("standardize", True):
        preprocessing_steps.append("横截面标准化使用总体标准差（ddof=0），中性化残差也重新标准化")
    else:
        preprocessing_steps.append("标准化关闭，中性化残差保留其尺度")
    preprocess_definition = "预处理设置：" + "；".join(preprocessing_steps) + f"。数值秩及常数判断容差为 {tolerance:g}。"
    if not preprocess.get("winsorize", True) and not preprocess.get("standardize", True):
        preprocess_definition += " raw 直接使用输入因子值，signal=value；不改变真实零值、负值或原始尺度。"
    else:
        preprocess_definition += " raw 表示未中性化版本，仍包含上述已开启的预处理，并非输入值原样通过。"
    sample_definition = (f"本次选择{selected_label}。各版本独立筛选所需输入；raw 只要求有效因子值，不受行业或 Barra 缺失影响。"
                         "中性化版本仅使用自身所需控制完整的股票，失败不会删除其它版本的信号。各版本收益差异可能同时来自处理方式和样本差异；控制字段交集仅作覆盖诊断。"
                         if _independent_samples(config) else
                         f"本次选择{selected_label}。每个因子的所选信号使用相同股票样本以便比较；这项比较规则不是原始等权回测对行业或市值的数学要求。不同因子的覆盖率可以不同。")
    items = [
        f"投资域为{universe}，包含金融行业；剔除信号时点已知 ST/*ST、上市不足 {minimum} 个交易日及入场停牌股票。",
        "每周首个交易日前一自然日日终形成信号；按该周首个交易日开盘至下一周首个交易日开盘计算持有期收益。",
        preprocess_definition,
        f"{neutral_definition}。如启用中性化，OLS 只对当期因子中性化，收益保持原始口径；控制缺失不补零，共线控制保留等价张成空间并记录。",
        sample_definition,
        "RankIC（Spearman）为主要指标，Pearson IC 附报。TOP/BOTTOM 按各套处理后因子的平均排名分位选择；并列值不拆分，子样本和十分组可以不等人数或为空。",
        "Q1 为低因子值，Q10 为高因子值，不根据历史收益翻转方向。每周入场按组内等权形成理论组合。",
        f"基准为过滤后{universe}等权组合，独立于因子或风险控制字段覆盖；每周收益按相同持有期计算。超额收益为逐周组合收益减基准收益，再进行复利。",
        f"本次年化频数为 {annualization}。总收益=期末净值−1；年化收益=期末净值^({annualization}/有效连续期数)−1；波动率为周收益样本标准差×√{annualization}。",
        "报告沿用附件的信息比率名称，其定义为几何年化收益/年化波动率；无无风险收益调整。超额信息比率采用同一公式。",
        f"IC 波动率使用样本标准差；年化 ICIR=IC均值/IC波动率×√{annualization}；普通 t=IC均值/IC波动率×√有效期数。IC 和收益胜率分别为 IC>0、收益>0 的比例。",
        f"普通 t 值没有修正序列相关，另列 Newey–West t 值及 95% 区间（配置滞后：{lag} 周）；累计 IC 为算术累加。",
        "Q10−Q1 为两腿周收益直接相减后复利，名义多空各 100%，不除以 2；它是理论多空差异，不包含融券及融资约束。",
        f"双边换手=目标权重与收益漂移后旧权重之差的绝对值之和；年化换手为周度双边换手均值×{annualization}。首期建仓计入，末期不假设清仓；多空换手为两腿之和。",
        "结果为费前理论收益及理论换手，不含手续费、税费、滑点和冲击成本；停牌和涨跌停限制另行披露，理论换手不等于可成交换手。",
        "退出停牌采用一致全收益口径的估值延续，中途退市计入终止收益。组内未解决收益使整组该期收益缺失，不能删除股票后重新归一化。",
        "空组及收益缺口不补零、不跨缺口拼接净值。各连续区段从 1 展示，方块表示区段起点，后续区段使用虚线；存在内部缺口时，全期间累计收益、年化收益和最大回撤为 N/A。",
        "最大回撤依据周度净值测量，不能反映周内最大回撤。此报告衡量因子排序及分组表现，不包含预测模型训练。",
    ]
    policy = config.get("data_policy", {})
    if policy.get("mode", "classic") in {"classic", "proxy"}:
        items[15] = ("退出停牌按配置沿用最近复权收盘估值；该估值不是成交价。退市或缺价无法解决时保留收益缺失，"
                     "不假定清算为零，也不删除股票后重新分配组内权重。")
        items.append("数据处理说明：" + str(policy.get("description", "")))
        unknown = policy.get("st_unknown", "include" if policy.get("mode", "classic") == "classic" else "error")
        items.append("历史 ST 未知时的处理：" + {"include": "保留股票并统计未知比例；不代表已确认非 ST。",
                     "exclude": "从股票池排除并报告未知比例。", "error": "停止构建股票池。"}[unknown])
        items.append("提供 GF 交易代理时，成交量为零或收盘为零用作停牌代理；成交量为当日完整数据，研究过滤含事后日内信息，"
                     "不能称为开盘前可知的可交易股票池。涨跌停比例估计仅用于交易限制诊断。")
        items.append("复权价格收益口径：" + str(config.get("price_contract", {}).get("description", "")))
        items.append("IC 仅在已冻结信号中使用有效收益配对，并提供标签覆盖率；任何缺失持仓标签使该组合当周收益缺失。")
        if policy.get("mode", "classic") == "classic":
            suspension = policy.get("suspension_unknown", "include")
            items.append("停牌状态缺失时的处理：" + {"include": "保留未知标记并纳入理论样本；仅有状态缺失、但价格有效时不作收益缺失。",
                         "exclude": "按入场日已冻结规则排除并披露覆盖率。", "error": "按显式配置停止构建股票池。"}[suspension])
            items.append("完整历史 ST、完整交易例外和逐笔公司行动不是因子统计的统一门槛；缺失终止收益只影响持有该股票的组合与对应标签。")
    if policy.get("missing_returns") == "available_case":
        items[15] = ("退出停牌按配置沿用最近复权收盘估值；该估值不是成交价。未知退市退出价值及无法解决的缺价保持缺失，"
                     "不向退市后延续价格，也不把未观察到的个股收益填成零。")
        items[16] = ("有效样本统计：先冻结排名、分组和目标权重 target_weight，再在原组内对有效收益计算统计权重 "
                     "evaluation_weight=target_weight/有效收益对应目标权重之和；缺失标签股票仍保留在持仓明细，统计权重为零。"
                     "因此统计收益可计算不代表全部持仓收益已知；统计权重与冻结持仓权重不同，可能产生非随机缺失导致的选择偏差。")
        items = [item for item in items if not item.startswith("IC 仅在已冻结信号中使用有效收益配对")]
        items.append("IC 仅用有效因子和收益配对，不填充缺失 IC；缺失收益不改变当期股票排名、分组或冻结目标权重。")
        items.append("独立基准采用相同的有效收益统计规则，并单列缺失标签及权重；超额收益仍为组收益减基准收益。")
        items.append("每周统计收益均可计算时，从初始净值 1 连续复利；非空组全无有效收益、整周信号缺失仍为 NA，"
                     "保留明确缺口，不能为了连线虚构收益。真实亏损归零后不重建本金。")
        items.append("漂移换手需要每只旧持仓的实际收益，缺失时仍为 NA，不使用统计收益代替；另列目标双边换手，"
                     "仅比较相邻冻结目标股票权重，不含收益漂移，不能冒充实际成交或漂移换手。")
        items.append("Q10−Q1 的统计权重合计为两腿毛权重之和，最高为 200%；现金比例和缺失权重为两腿比例的平均值，"
                     "分母不同，不能将这些字段相加解释为单个多头账户的资产配置。")
    if policy.get("empty_groups") == "cash":
        if policy.get("missing_returns") != "available_case":
            items[16] = ("无法计算的非空组收益不补零、不跨缺口拼接净值。各连续区段从 1 展示；"
                         "存在内部缺口时，全期间累计收益、年化收益和最大回撤为 N/A。")
        items.append("空组现金约定：当周整体信号有效但某分组因并列值为空时，该组以现金为目标，周收益为 0；"
                     "这是一项显式组合约定。整周无信号或非空组全部缺失收益不属于空组现金，不补零。")
    if "full" in variants:
        items.append("全中性使用的风险暴露：" + "、".join(str(field) for field in risk_fields) + "。")
    if backtest.get("start_date") or backtest.get("end_date"):
        items.append(f"配置的信号入场日期范围：{backtest.get('start_date') or '不限'} 至 {backtest.get('end_date') or '不限'}；实际首末持有期见样本信息。")
    items.append("因子更新频率与调仓频率分别确定；每次调仓取截点前最近一条记录，不对因子做周/月平均。date 必须是实际可用日期，最新记录为空时不越过它寻找旧值。")
    if monthly:
        items.append("本次按月调仓：完整交易日历中每月首个交易日开盘进入，持有至下月首个交易日开盘；月中开始的区间不会伪造月初调仓日。")
        items.append("periods.parquet、ic_periodic.csv、portfolio_periodic.csv 以持有期为单位；兼容文件 weeks.parquet 及 *_weekly.csv 在本次也存月度记录，文件名不改变收益频率。")
    sources = ""
    if policy.get("missing_returns") == "available_case":
        sources = ('<p class="caption">方法参考：<a href="https://github.com/microsoft/qlib/blob/main/qlib/contrib/eva/alpha.py">'
                   'Qlib calc_long_short_return</a> 默认先按预测信号选组，再对有效 label 求均值；'
                   '<a href="https://github.com/stefan-jansen/alphalens-reloaded/blob/main/src/alphalens/utils.py">'
                   'Alphalens get_clean_factor</a> 则先删除不完整的因子与收益记录再分组。'
                   '本项目采用分组后有效收益统计，保留原始分组和缺失披露；空组现金是本项目配置约定。'
                   '<a href="https://github.com/microsoft/qlib/blob/main/qlib/backtest/account.py">Qlib 持仓估值</a>'
                   '在停牌时保留已有价格，这与统计均值计算属于不同层次；本报告不是持续现金账户的可实现全收益。</p>')
    return _contextualize(_details("指标定义与回测口径", "<ol>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ol>" + sources, True), config)


def _prioritize_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Put exceptional records before routine successful preprocessing rows."""
    if frame.empty:
        return frame
    result = frame.copy()
    status = result.get("status", pd.Series("", index=result.index)).fillna("").astype(str).str.lower()
    redundant = result.get("redundant_columns", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().ne("")
    failed = ~status.isin(["", "ok", "constant"])
    result["_report_priority"] = np.select([failed, status.eq("constant"), redundant], [0, 1, 2], default=3)
    order = ["_report_priority"] + [col for col in ["entry", "factor", "variant"] if col in result]
    return result.sort_values(order, kind="stable").drop(columns="_report_priority")


def _portfolio_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio in PORTFOLIOS:
        block = frame[frame.portfolio.eq(portfolio)]
        if block.empty:
            continue
        counts = pd.to_numeric(block.n_stocks, errors="coerce")
        returns = pd.to_numeric(block["return"], errors="coerce")
        restricted = pd.to_numeric(block.get("trade_restricted_fraction", pd.Series(np.nan, index=block.index)), errors="coerce")
        reasons = block.get("reason", pd.Series("", index=block.index)).fillna("").astype(str)
        rows.append({"portfolio": portfolio, "calendar_weeks": len(block),
                     "valid_return_weeks": int(np.isfinite(returns).sum()),
                     "empty_weeks": int((counts.eq(0) | reasons.eq("empty_leg")).sum()),
                     "missing_return_weeks": int((~np.isfinite(returns)).sum()),
                     "mean_stocks": counts.mean(), "min_stocks": counts.min(), "max_stocks": counts.max(),
                     "mean_restricted_fraction": restricted.mean(), "max_restricted_fraction": restricted.max()})
        if "data_complete" in block:
            rows[-1]["incomplete_label_weeks"] = int((~_true_flags(block.data_complete) & counts.gt(0)).sum())
        if "missing_weight" in block:
            missing_weight = pd.to_numeric(block.missing_weight, errors="coerce")
            rows[-1].update(mean_missing_weight=missing_weight.mean(), max_missing_weight=missing_weight.max())
        if "evaluation_weight_sum" in block:
            rows[-1]["mean_evaluation_weight"] = pd.to_numeric(block.evaluation_weight_sum, errors="coerce").mean()
        if "cash_weight" in block:
            rows[-1]["mean_cash_weight"] = pd.to_numeric(block.cash_weight, errors="coerce").mean()
    return pd.DataFrame(rows)


def _artifact_links(root: Path, relative: str = "") -> str:
    links = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}:
            # Optional account files may belong to an earlier run. They are
            # linked only by the enabled, validated execution section below.
            if path.name.startswith("execution_"):
                continue
            target = relative + path.name
            links.append(f'<li><a href="{_escape(target)}">{_escape(path.name)}</a></li>')
    return "<h2>机器可读产物</h2><ul>" + "".join(links) + "</ul>"


def _true_flags(values: pd.Series) -> pd.Series:
    return values.fillna(False).astype(str).str.lower().isin({"true", "1", "1.0"})


def _execution_nav_points(frame: pd.DataFrame) -> list[dict]:
    """Normalize a persistent cash account without rebasing it after a gap."""
    block = frame.sort_values("entry")
    if block.empty:
        return []
    initial = _number(block.nav_start.iloc[0])
    if initial is None or initial <= 0:
        return []
    first = _date(block.entry.iloc[0])
    points = [{"date": first, "value": 1., "segment": 0}]
    previous_exit = first
    for row in block.to_dict("records"):
        entry, exit_date = _date(row.get("entry")), _date(row.get("exit"))
        if entry != previous_exit:
            points.append({"date": entry, "value": None})
        nav = _number(row.get("nav"))
        points.append({"date": exit_date, "value": nav / initial if nav is not None else None, "segment": 0})
        previous_exit = exit_date
    return points


def _execution_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for portfolio, block in frame.groupby("portfolio", sort=False):
        numeric = lambda name: pd.to_numeric(block[name], errors="coerce")
        missing = _true_flags(block.get("signal_missing", pd.Series(False, index=block.index)))
        rows.append({"portfolio": "ALL（共同样本多头）" if portfolio == "ALL" else portfolio,
                     "rebalance_weeks": len(block), "mean_stocks": numeric("n_stocks").mean(),
                     "mean_cash_weight": numeric("cash_weight").mean(),
                     "max_cash_weight": numeric("cash_weight").max(),
                     "target_empty_weeks": int((_true_flags(block.target_empty) & ~missing).sum()),
                     "signal_missing_weeks": int(missing.sum()),
                     "blocked_buy_weeks": int(numeric("blocked_buy_value").gt(0).sum()),
                     "blocked_sell_weeks": int(numeric("blocked_sell_value").gt(0).sum()),
                     "blocked_buy_value": numeric("blocked_buy_value").sum(min_count=1),
                     "blocked_sell_value": numeric("blocked_sell_value").sum(min_count=1),
                     "carried_valuation_count": numeric("carried_valuation_count").sum(min_count=1)})
    return pd.DataFrame(rows)


def _execution_section(folder: Path, factor: dict, charts: _Charts, config: dict | None = None) -> str:
    """Display account results only when explicitly enabled by this run's card."""
    heading = '<h2 id="execution">执行账本（可选）</h2>'
    metadata = factor.get("execution", {})
    if isinstance(metadata, dict) and metadata.get("status") == "skipped":
        reasons = metadata.get("blocking", [])
        return (heading + _notice("可选执行账本已跳过，因子统计已独立完成", "附加成交模拟的数据未满足其记账要求；以下原因仅影响执行账本，不阻断 IC、理论分组收益及换手。")
                + "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in reasons) + "</ul>")
    if not isinstance(metadata, dict) or metadata.get("enabled") is not True:
        return heading + '<p class="muted">本次未启用执行账本。可选复权单位代理账本用于交易限制敏感性；原始股数账本则另需完整公司行动与有效退出数据。上方权重分组与 IC 已独立计算。</p>'
    weekly, performance = _csv(folder / "execution_weekly.csv"), _csv(folder / "execution_performance.csv")
    weekly_required = {"entry", "exit", "variant", "portfolio", "nav_start", "return", "nav", "turnover",
                       "cash_weight", "n_stocks", "blocked_buy_value", "blocked_sell_value",
                       "carried_valuation_count", "target_empty"}
    performance_columns = PERFORMANCE_COLUMNS[:9] + ["status"]
    performance_required = set(performance_columns) | {"variant"}
    problems = []
    if weekly.empty or not weekly_required <= set(weekly):
        problems.append("execution_weekly.csv 缺失、为空或字段不完整。")
    if performance.empty or not performance_required <= set(performance):
        problems.append("execution_performance.csv 缺失、为空或字段不完整。")
    if problems:
        return heading + _notice("执行账本产物尚不完整", " ".join(problems) + " 本节不展示旧执行结果；上方理论因子测试保持独立。")
    if weekly.duplicated(["entry", "exit", "variant", "portfolio"]).any() or performance.duplicated(["variant", "portfolio"]).any():
        return heading + _notice("执行账本产物未通过核验", "账户周次或绩效汇总存在重复键，本节未展示执行数值。")
    portfolios = ["ALL"] + PORTFOLIOS[:-1]
    variants = _variant_labels(config, values=weekly.variant.unique())
    supported = weekly.variant.isin(variants) & weekly.portfolio.isin(portfolios)
    sections = [heading, _notice("现金与持股逐日记账的费前执行模拟", "本节账户保留旧持仓和现金，并处理交易限制。Q10、Q1 是各自独立的多头账户，不声称实际融券多空；主理论分组检验保持原口径。", "info")]
    adjusted_units = metadata.get("price_mode") == "adjusted_units"
    if adjusted_units:
        sections = [heading, _notice("复权单位与现金的交易限制代理", "持仓数量表示复权价格单位，不是真实股数。"
                    "修复后的复权价格已含其供应商调整，不重复计入现金分红或送转。退市退出价值未解决的账户从受影响周起停止收益展示，其他账户继续。", "info")]
    if not supported.all():
        sections.append(_notice("部分执行条目未展示", "执行章节只接受支持的信号版本及其 ALL 与 Q1—Q10 独立多头账户；其它条目未纳入图表和汇总。"))
    weekly = weekly[supported].copy()
    performance = performance[performance.variant.isin(variants) & performance.portfolio.isin(portfolios)].copy()
    pairs = set(zip(weekly.variant, weekly.portfolio))
    if not pairs or pairs != set(zip(performance.variant, performance.portfolio)):
        return heading + _notice("执行账本产物未通过核验", "周度账本与绩效表的账户集合不一致，本节未展示执行数值。")
    for column in ["entry", "exit"]:
        weekly[column] = pd.to_datetime(weekly[column], errors="coerce")
    invalid_dates = (weekly[["entry", "exit"]].isna().any().any() or (weekly.exit <= weekly.entry).any()
                     or weekly.duplicated(["entry", "exit", "variant", "portfolio"]).any())
    initial = pd.to_numeric(weekly.sort_values("entry").groupby(["variant", "portfolio"], sort=False).head(1).nav_start, errors="coerce")
    if invalid_dates or any(not _finite(value) or value <= 0 for value in initial):
        return heading + _notice("执行账本产物未通过核验", "持有日期或账户初始资产无效，不能归一化执行净值。")
    rules = [
        "停牌同时禁止买卖；涨停禁止买入、跌停禁止卖出；交易状态未知或开盘价缺失时不下可成交订单，旧持仓继续保留。",
        "每周开盘先处理当日公司行动、终止结算和估值，再先卖后买；卖不出的旧持仓保留，现金不足时按同一比例缩减可执行买单，不借款。",
        "分红、送转基于原始未复权价格显式入账：现金分红在除息事件日进入现金，未模拟实际派息到账日差异；送转增加股数，历史估值同步作公司行动调整。",
        "只在每周调仓开盘执行；未成交订单不每日追单，下次调仓重新计算。有效空目标尝试卖出旧仓；缺失信号则冻结股数和现金，不把它当作空目标清仓。",
        "允许分数股；现金不计息，无费税、滑点及冲击成本。ALL 是该因子共同样本的独立多头账户，与上方独立股票池基准分开统计。",
        "绩效表仍按周度开盘到开盘收益汇总，最大回撤来自周度账户净值；每日持仓估值保存在 execution_daily.parquet，末个退出日记录开盘，其余日记录收盘。",
        "未成交买入金额同时包含交易限制及现金不足导致的缩买；未成交金额为各周请求与实际成交差额之和，不能当作单一时点资金缺口。沿用估值次数为股票×估值时点累计次数，不是不同停牌股票数量。",
    ]
    if adjusted_units:
        rules[1] = "每周开盘先按复权价格估值，再先卖后买；受阻旧仓保留，现金不足按同一比例缩减允许的买单，不借款。"
        rules[2] = "股数以复权单位表示，不叠加原始股本口径的分红、送转或退市现金；这是价格收益与交易限制的敏感性模拟。"
        rules[4] = "允许分数复权单位；现金不计息，无费税、滑点和冲击成本。ALL 是共同样本的独立多头账户。"
        rules.append("quality_status 标识账户数据缺口；indicative_nav 仅保留无法验证退出时的账面追溯值，不计入收益或绩效。")
    sections.append(_details("执行模拟规则与账本口径", "<ul>" + "".join(f"<li>{_escape(rule)}</li>" for rule in rules) + "</ul>", True))
    if any(key != "enabled" for key in metadata):
        sections.append(_details("本次执行概要", _records({key: value for key, value in metadata.items() if key != "enabled"})))
    for variant, title in variants.items():
        block = weekly[weekly.variant.eq(variant)]
        if block.empty:
            continue
        nav_series, cash_series = [], []
        for portfolio in portfolios:
            account = block[block.portfolio.eq(portfolio)].sort_values("entry")
            if account.empty:
                continue
            label = "ALL（共同样本多头）" if portfolio == "ALL" else portfolio
            nav_series.append({"name": label, "points": _execution_nav_points(account)})
            cash_series.append({"name": label, "points": _points(account, "cash_weight", "entry")})
        sections.append(f"<h3>{_escape(title)}：执行账户</h3>")
        sections.append(charts.chart(f"{title}：执行账户净值", nav_series,
                                     caption="每个账户用首期交易前资产归一为 1，随后保留真实账户状态；缺口断线但不重置本金。净值点为下一周调仓前开盘估值。"))
        sections.append(charts.chart(f"{title}：执行后现金权重", cash_series, unit="percent",
                                     caption="日期为当周入场日，权重取该日开盘调仓后；买不进的资金保留为现金，卖不出的股票继续持有。"))
        selected = performance[performance.variant.eq(variant)].copy()
        selected["portfolio"] = pd.Categorical(selected.portfolio, portfolios, ordered=True)
        selected = selected.sort_values("portfolio")
        selected["portfolio"] = selected.portfolio.astype(object).replace({"ALL": "ALL（共同样本多头）"})
        sections.append(render_table(selected, columns=performance_columns))
        sections.append(_details("执行限制、现金及沿用估值统计", render_table(_execution_diagnostics(block), 20), True))
    files = ["execution_weekly.csv", "execution_performance.csv", "execution_holdings.parquet", "execution_trades.parquet", "execution_daily.parquet"]
    links = [f'<li><a href="{filename}">{filename}</a></li>' for filename in files if (folder / filename).is_file()]
    sections.append("<h3>执行账本产物</h3><ul>" + "".join(links) + "</ul>")
    return _contextualize("\n".join(sections), config or {})


_CSS = """
:root{color-scheme:light;font-family:system-ui,-apple-system,'Microsoft YaHei',sans-serif;color:#18293b;background:#f3f6fa}
*{box-sizing:border-box}body{max-width:1440px;margin:0 auto;padding:32px 24px;font-size:14px;line-height:1.7}h1{font-size:30px}h2{margin-top:38px;font-size:23px}h3{font-size:17px;margin:8px 0 10px}a{color:#155e9b}p{margin:8px 0}section,details,.card{background:#fff;border:1px solid #dce4ed;border-radius:9px;padding:18px;margin:14px 0}.notice{border-left:5px solid #b77e1f;background:#fff3d5;padding:16px 20px;margin:18px 0;border-radius:4px}.notice.info{background:#e5f3fc;border-color:#155e9b}.notice.synthetic{background:#fff0f4;border-color:#b82f60}.notice strong{font-size:18px}.muted,.caption{font-size:12px;color:#5c6c7c}.caption{min-height:1.6em}details summary{cursor:pointer;font-weight:600;font-size:16px}details[open] summary{margin-bottom:15px}.table-wrap{overflow:auto;max-width:100%;margin:12px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;border-bottom:1px solid #dce4ed;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#edf3f9;font-weight:600}tbody tr:hover{background:#f7faff}.grid-two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.grid-two .chart{margin:0}.chart{min-width:0}.plot{position:relative}.plot svg{display:block;width:100%;height:auto;touch-action:pan-y}.plot text{font-size:11px;fill:#647487}.grid{stroke:#e4ebf2;stroke-width:1}.zero{stroke:#9babbc;stroke-width:1}.legends{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:7px}.legend{border:1px solid #dbe4ed;background:#fff;border-radius:4px;padding:4px 8px;color:#32485b;cursor:pointer;font-size:12px}.legend:before{content:'';display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px;background:var(--series-color)}.legend[aria-pressed=false]{opacity:.4}.tooltip{position:absolute;z-index:2;pointer-events:none;background:rgba(17,32,47,.95);color:#fff;padding:10px 13px;border-radius:6px;font-size:12px;max-height:270px;overflow:hidden;min-width:140px;box-shadow:0 3px 15px #0002}.tooltip table{width:auto}.tooltip td{padding:1px 5px;border:0;font-size:12px}.crosshair{stroke:#738699;stroke-dasharray:3 3;pointer-events:none}.nav{display:flex;gap:16px;flex-wrap:wrap}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.cards .card{margin:0}.mono{font-family:ui-monospace,monospace;overflow-wrap:anywhere}footer{margin-top:35px;padding-top:15px;border-top:1px solid #dce4ed;color:#5c6c7c;font-size:12px}@media(max-width:850px){body{padding:20px 12px}.grid-two{grid-template-columns:1fr}h1{font-size:24px}section,details{padding:12px}}@media print{body{background:white;max-width:none}.legend{border:0}.tooltip{display:none!important}details{break-inside:avoid}.chart{break-inside:avoid}a{color:inherit}}
"""
_JS = """
document.querySelectorAll('.chart-data').forEach(node=>{
 const chart=node.closest('.chart'),svg=chart.querySelector('svg'),tip=chart.querySelector('.tooltip'),cross=svg.querySelector('.crosshair');
 const data=JSON.parse(node.textContent),hidden=new Set();
 data.series.forEach(s=>{s.lookup=new Map(s.points.map(p=>[p.d,p]));});
 chart.querySelectorAll('.legend').forEach(button=>button.addEventListener('click',()=>{
  const i=Number(button.dataset.series);hidden.has(i)?hidden.delete(i):hidden.add(i);button.setAttribute('aria-pressed',String(!hidden.has(i)));
  svg.querySelectorAll('[data-series="'+i+'"]').forEach(el=>{el.style.display=hidden.has(i)?'none':'';});tip.hidden=true;
 }));
 svg.addEventListener('pointermove',event=>{
  const pt=svg.createSVGPoint();pt.x=event.clientX;pt.y=event.clientY;const local=pt.matrixTransform(svg.getScreenCTM().inverse());
  if(local.x<74||local.x>940||local.y<10||local.y>275){tip.hidden=true;cross.setAttribute('visibility','hidden');return;}
  let lo=0,hi=data.axis.length-1;while(lo<hi){const mid=(lo+hi)>>1;if(data.axis[mid].x<local.x)lo=mid+1;else hi=mid;}
  let index=lo;if(index>0&&Math.abs(data.axis[index-1].x-local.x)<Math.abs(data.axis[index].x-local.x))index--;
  const nearest=data.axis[index];tip.replaceChildren();const heading=document.createElement('strong');heading.textContent=nearest.d;tip.append(heading);
  const table=document.createElement('table');let count=0;
  data.series.forEach((s,i)=>{if(hidden.has(i))return;const p=s.lookup.get(nearest.d);if(!p)return;count++;
   const row=document.createElement('tr'),name=document.createElement('td'),value=document.createElement('td');name.textContent=s.name;name.style.color=s.color;
   value.textContent=(data.unit==='percent'?(p.v*100).toFixed(3)+'%':p.v.toFixed(5))+(p.reset?'（区段起点）':'');row.append(name,value);table.append(row);
  });tip.append(table);if(!count){const empty=document.createElement('p');empty.textContent='缺失 / 无有效值';tip.append(empty);}
  tip.hidden=false;cross.setAttribute('x1',nearest.x);cross.setAttribute('x2',nearest.x);cross.setAttribute('visibility','visible');
  const box=svg.getBoundingClientRect(),parent=chart.querySelector('.plot').getBoundingClientRect();
  const offset=event.clientX-parent.left;tip.style.left=Math.max(0,Math.min(offset+12,parent.width-tip.offsetWidth-5))+'px';tip.style.top=Math.max(0,Math.min(event.clientY-box.top+8,parent.height-tip.offsetHeight))+'px';
 });svg.addEventListener('pointerleave',()=>{tip.hidden=true;cross.setAttribute('visibility','hidden');});
});
"""


def _write(path: Path, title: str, sections: list[str]) -> Path:
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{_escape(title)}</title><style>{_CSS}</style></head><body>'
                + "\n".join(sections)
                + '<footer>所有图表和交互脚本均内嵌，可离线打开。点击图例隐藏曲线，移动指针查看数值；缺失值显示为“—”，不代表零。</footer>'
                + f'<script>{_JS}</script></body></html>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def _synthetic(config: dict) -> str:
    if config.get("dataset_kind") == "synthetic":
        return _notice("合成数据验收示例", "仅验证工程流程；以下数值均为模拟数据，不是 /data/mongo 的真实因子实证，不构成因子有效性结论。", "synthetic")
    return ""


def _proxy_notice(config: dict) -> str:
    mode = config.get("data_policy", {}).get("mode", "classic")
    if mode == "classic":
        return _notice("经典因子统计口径", "因子、时点及价格支持核心统计；ST 与交易状态采用已有历史记录和代理，未知保持披露。"
                       "完整历史 ST、特殊交易例外、公司行动明细和退市现金不作为统一数据门槛。缺失收益保留并影响相应配对或组合；"
                       "可选执行账本单独核验，结果是费前因子诊断。", "info")
    if mode != "proxy":
        return ""
    return _notice("代理口径的因子研究", "使用已披露的 ST、停牌和涨跌停代理。未知状态与缺失收益有单独记录；"
                   "收益为核验后的复权价格收益，未声明覆盖全部公司行动或真实退市兑付。结果用于因子诊断，不能视为实际可成交的净收益。")


def _quality_disclosure(config: dict, audit: dict, manifest: dict, usable: bool) -> str:
    """Keep readiness and data completeness separate, without relaxing integrity checks."""
    if config.get("data_policy", {}).get("mode", "classic") != "classic":
        return ""
    rows = [{"项目": "核心因子统计", "本次状态": "已完成" if usable else "尚未完成", "影响":
             "可选筛选和执行数据缺项不会自动中止核心统计；时点、键、价格及产物一致性仍须核验。"}]
    available_case = config.get("data_policy", {}).get("missing_returns") == "available_case"
    if available_case:
        rows.append({"项目": "缺失收益统计规则", "本次状态": "冻结分组后按有效收益统计", "影响":
                     "目标权重保持不变；统计权重在有效标签内归一，可能产生选择偏差。收益可计算与标签完整分别记录。"})
    if manifest.get("status") == "ready":
        base_count = manifest.get("base_rows", 0)
        unknown_st = manifest.get("unknown_st_base_rows")
        if _finite(unknown_st) and _finite(base_count) and base_count > 0:
            rows.append({"项目": "入选股周 ST 状态未知", "本次状态": f"{int(unknown_st):,} / {int(base_count):,}（{unknown_st / base_count:.2%}）",
                         "影响": "依据冻结配置纳入或排除；未知不等于正常状态，不宣称准确剔除所有历史 ST。"})
        if _finite(manifest.get("unknown_suspension_base_rows")):
            rows.append({"项目": "入选股周停牌状态未知", "本次状态": f"{int(manifest['unknown_suspension_base_rows']):,}",
                         "影响": "保留未知标记；价格有效时可计算理论收益，价格缺失时单独披露，不把状态未知当成正常成交。"})
        if _finite(manifest.get("label_coverage")):
            rows.append({"项目": "未来收益标签覆盖率", "本次状态": f"{manifest['label_coverage']:.5%}",
                         "影响": "IC 使用已冻结样本中的有效配对，缺失标签不改变当期分组及目标权重。"})
        if _finite(manifest.get("unresolved_labels")):
            rows.append({"项目": "未解决收益股周", "本次状态": f"{int(manifest['unresolved_labels']):,}",
                         "影响": ("原组保留股票及缺失标记，使用其余有效收益统计；非空组全部缺失时仍为 NA，未观察个股收益不补零。" if available_case else
                                   "受影响组合当周收益为缺失，未受影响组合和有效 IC 继续计算；不以零收益替代。")})
        if _finite(manifest.get("incomplete_benchmark_weeks")):
            rows.append({"项目": "基准收益无法计算周数", "本次状态": f"{int(manifest['incomplete_benchmark_weeks']):,}",
                         "影响": "这些周次的基准和相应超额收益缺失；组合自身收益可计算时仍可展示。"})
        if _finite(manifest.get("partial_benchmark_weeks")):
            rows.append({"项目": "基准标签不完整但收益可计算周数", "本次状态": f"{int(manifest['partial_benchmark_weeks']):,}",
                         "影响": "有效标签按相同统计规则汇总；缺失股票及目标权重另报，连续基准净值不代表标签已全部完整。"})
    execution = config.get("execution", {})
    requested = execution.get("enabled", False)
    execution_passed = audit.get("execution_gate_passed", True)
    rows.append({"项目": "可选执行账本", "本次状态": "未启用" if not requested else ("附加数据核验通过" if execution_passed else "附加数据未就绪，已跳过"),
                 "影响": "执行账本与理论分组统计分别披露；复权单位代理与原始股数现金记账使用不同数据要求。"})
    body = "<h2>计算状态与数据完整性</h2>" + render_table(pd.DataFrame(rows))
    if requested and not execution_passed:
        body += _details("可选执行账本未运行的原因", "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in audit.get("execution_blocking", [])) + "</ul>")
    return _contextualize(body, config)


def _blocked(issues: list[str]) -> str:
    return (_notice("尚无正式研究结论", "数据、面板或执行门槛尚未通过，或者已有结果不属于当前冻结配置。未展示旧结果，不能据此判断因子有效或无效。")
            + "<h2>待补齐或待解决</h2><ul>" + "".join(f"<li>{_escape(item)}</li>" for item in issues) + "</ul>")


def _factor_path(root: Path, factor: dict) -> Path | None:
    name = str(factor.get("name", ""))
    if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        return None
    expected = (root / "factors" / name).resolve()
    declared = (root / str(factor.get("output_dir", f"factors/{name}"))).resolve()
    return expected if declared == expected and expected.parent == (root / "factors").resolve() else None


def _factor_report(root: Path, folder: Path, factor: dict, config: dict, audit: dict, coverage: pd.DataFrame, usable: bool, issues: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    name, label = str(factor.get("name", folder.name)), str(factor.get("label", factor.get("name", folder.name)))
    sections = [_synthetic(config), _proxy_notice(config), '<nav class="nav"><a href="../../report.html">← 测试汇总与审计</a></nav>',
                f"<h1>{_escape(label)} 因子测试报告</h1><p class=\"mono\">{_escape(name)}</p>"]
    frames = {}
    if usable:
        required = {
            "ic_weekly": {"entry", "exit", "variant", "method", "subset", "n_stocks", "ic", "cumulative_ic"},
            "ic_summary": set(IC_COLUMNS) | {"variant", "method"},
            "portfolio_weekly": {"entry", "exit", "variant", "portfolio", "n_stocks", "return", "benchmark_return", "excess_return", "turnover", "nav", "excess_nav", "segment_id", "complete"},
            "performance": set(PERFORMANCE_COLUMNS) | {"variant"},
        }
        for filename, columns in required.items():
            frame = _csv(folder / f"{filename}.csv")
            frames[filename] = frame
            if not columns <= set(frame.columns):
                issues = issues + [f"{name}/{filename}.csv 缺失、不可读或字段不完整。"]
                usable = False
    if not usable:
        sections += [_blocked(issues), _definitions(config), _artifact_links(folder)]
        return _write(folder / "report.html", f"{label} · 数据门槛", [_contextualize(section, config) for section in sections])
    charts = _Charts()
    sample_notice = ("各版本独立筛选样本，raw 不受行业或 Barra 缺失影响；不同版本的结果差异同时可能包含样本差异。"
                     if _independent_samples(config) else "所选信号共享样本；")
    sections.append(_notice("费前因子分组与相关性检验", sample_notice + "不训练预测模型。理论等权组合及双边换手不包含交易费用，也不等于实际可成交结果。", "info"))
    sections.append(_quality_disclosure(config, audit, load_json(root / "panel_manifest.json"), True))
    sections.append("<h2>测试信息</h2>" + _sample_records(factor.get("sample", {}), config))
    if factor.get("summary"):
        sections.append(_records(factor["summary"]))
    sections.append(_definitions(config))
    sections.append("<h2>1. 覆盖度</h2>")
    local = coverage[coverage.factor.astype(str) == name].sort_values("entry") if "factor" in coverage else pd.DataFrame()
    coverage_title = "逐周因子覆盖与各版本测试样本" if _independent_samples(config) else "逐周因子覆盖与所选信号共同样本"
    sections.append(charts.chart(coverage_title, _coverage_series(local, config), unit="percent",
                                caption=f"分母为独立的过滤后{_universe_label(config)}基准股票池。控制字段缺失不会被填为零。"))
    sections.append(_details("覆盖、零值及并列情况（完整数据见 CSV）", render_table(local, 30)))
    for index, (variant, title) in enumerate(_variant_labels(config).items(), 2):
        sections.append(f'<h2 id="{variant}">{index}. {_escape(title)}</h2>')
        summary = frames["ic_summary"]
        primary = summary[(summary.variant == variant) & (summary.method == "spearman")].copy()
        primary["subset"] = pd.Categorical(primary.subset, SUBSETS, ordered=True)
        sections.append(f"<h3>{index}.1 RankIC 回测（主指标）</h3>" + render_table(primary.sort_values("subset"), columns=IC_COLUMNS))
        weekly = frames["ic_weekly"]
        main = weekly[(weekly.variant == variant) & (weekly.method == "spearman") & (weekly.subset == "ALL")].sort_values("entry")
        if "label_coverage" in main:
            sections.append(_details("收益配对覆盖率", render_table(main, 20,
                ["entry", "n_stocks", "n_labels", "label_coverage"])))
        sections.append('<div class="grid-two">' + charts.chart("周度 RankIC", [{"name": "RankIC", "points": _points(main, "ic")}], kind="bar", unit="percent", caption="横截面 Spearman 相关；日期为收益实现的退出日。")
                        + charts.chart("RankIC 时序累计", [{"name": "累计 RankIC", "points": _points(main, "cumulative_ic")}], caption="IC 为算术累加，不是投资净值。缺失周次断线。") + "</div>")
        pearson = summary[(summary.variant == variant) & (summary.method == "pearson")].copy()
        pearson["subset"] = pd.Categorical(pearson.subset, SUBSETS, ordered=True)
        pweekly = weekly[(weekly.variant == variant) & (weekly.method == "pearson") & (weekly.subset == "ALL")].sort_values("entry")
        sections.append(_details("Pearson IC 附加检验", render_table(pearson.sort_values("subset"), columns=IC_COLUMNS)
                                 + '<div class="grid-two">' + charts.chart("周度 Pearson IC", [{"name": "Pearson IC", "points": _points(pweekly, "ic")}], kind="bar", unit="percent")
                                 + charts.chart("Pearson IC 时序累计", [{"name": "累计 Pearson IC", "points": _points(pweekly, "cumulative_ic")}]) + "</div>"))
        sections.append(f"<h3>{index}.2 十分组回测</h3>")
        portfolio = frames["portfolio_weekly"]
        portfolio = portfolio[portfolio.variant == variant]
        nav, excess = [], []
        for group in PORTFOLIOS[:-1]:
            group_frame = portfolio[portfolio.portfolio == group]
            nav.append({"name": group, "points": _nav_points(group_frame, "nav")})
            excess.append({"name": group, "points": _nav_points(group_frame, "excess_nav")})
        nav_caption = "每周组内等权，净值起点为 1；方块表示重新归一的连续区段，后续区段为虚线。空组或收益缺口断线，未跨缺口拼接。"
        if config.get("data_policy", {}).get("missing_returns") == "available_case":
            nav_caption = ("冻结分组后按有效收益计算统计均值，每周收益可计算时从 1 连续复利。"
                           "缺失个股收益不补零，统计权重可能不同于冻结目标权重；非空组全缺或整周无信号仍显示缺口。")
            if config.get("data_policy", {}).get("empty_groups") == "cash":
                nav_caption += "整体信号有效时的空组按现金收益 0 处理。"
        sections.append(charts.chart("十分组组合表现", nav, caption=nav_caption))
        sections.append(charts.chart("十分组组合超额表现", excess, caption="逐周组合收益减配置股票池的等权基准收益再复利；超额净值不是组合净值与基准净值的比值。" + nav_caption))
        spread = portfolio[portfolio.portfolio == "Q10-Q1"]
        sections.append(charts.chart("Q10−Q1 理论多空组合净值", [{"name": "Q10−Q1", "points": _nav_points(spread, "nav")}], caption="每周收益为 Q10−Q1，名义多空各 100%，不除以 2；理论多空结果不代表可融券的资金账户。"))
        performance = frames["performance"]
        performance = performance[(performance.variant == variant) & performance.portfolio.isin(PORTFOLIOS)].copy()
        performance["portfolio"] = pd.Categorical(performance.portfolio, PORTFOLIOS, ordered=True)
        performance = performance.sort_values("portfolio")
        bars, exbars = [], []
        for group in PORTFOLIOS[:-1]:
            row = performance[performance.portfolio == group]
            bars.append({"date": group, "value": _number(row.annual_return.iloc[0]) if len(row) else None})
            exbars.append({"date": group, "value": _number(row.excess_annual_return.iloc[0]) if len(row) else None})
        sections.append('<div class="grid-two">' + charts.chart("十分组年化收益", [{"name": "年化收益", "points": bars}], kind="category", unit="percent")
                        + charts.chart("十分组超额年化收益", [{"name": "超额年化收益", "points": exbars}], kind="category", unit="percent") + "</div>")
        performance_columns = PERFORMANCE_COLUMNS.copy()
        if "annual_target_turnover" in performance:
            performance_columns.insert(performance_columns.index("annual_turnover") + 1, "annual_target_turnover")
        sections.append(render_table(performance, columns=performance_columns))
        sections.append(_details("分组股票数量、空组与交易受限诊断", render_table(_portfolio_diagnostics(portfolio), 20)
                                 + '<p class="caption">交易受限比例为当周组内股票数占比的周度均值及单周最高值；这不是实际未成交率。Q10−Q1 股票数为两腿股票数之和。共同样本 ALL 篮子未混入十分组表。</p>', True))
        if "missing_weight" in portfolio:
            sections.append(_details("缺失标签权重与统计口径逐周明细", render_table(portfolio, 30,
                ["entry", "portfolio", "n_stocks", "n_labels", "label_coverage", "data_complete", "missing_weight",
                 "evaluation_weight_sum", "cash_weight", "return_method", "turnover", "target_turnover"])
                + '<p class="caption">缺失标签目标权重按冻结持仓计算。统计权重仅用于有效样本均值，不能反推实际调仓；漂移换手缺失仍保持 NA。</p>'))
        annual = _csv(folder / "annual_performance.csv")
        if not annual.empty and "variant" in annual:
            selected = annual[(annual.variant == variant) & annual.portfolio.isin(PORTFOLIOS)]
            sections.append(_details("年度分组与多空表现", render_table(selected, 300, ["year"] + performance_columns)))
    sections.append(_execution_section(folder, factor, charts, config))
    sections.append("<h2>失败、缺口与交易限制</h2>")
    failures = _csv(root / "preprocessing_failures.csv")
    if not failures.empty and "factor" in failures:
        failures = failures[failures.factor.astype(str) == name]
    sections.append('<p class="caption">优先显示失败、常数信号及冗余控制记录，再显示常规正常记录；完整明细见 CSV。</p>' + render_table(_prioritize_diagnostics(failures), 60))
    sections.append("<p>停牌、退市估值和交易受限明细见根目录股票池与收益面板、基准及诊断产物；未知收益不会静默补零。</p>")
    sections = [_contextualize(section, config) for section in sections]
    sections.append(_details("数据审计与来源版本", _source_audit(audit)))
    sections.append(_artifact_links(folder))
    return _write(folder / "report.html", f"{label} 因子测试报告", sections)


def report(run_dir: str | Path) -> Path:
    """Write the run index and current factor reports; never display stale results."""
    requested = Path(run_dir).resolve()
    if not requested.is_dir():
        raise ValueError(f"Run directory does not exist: {requested}")
    if requested.parent.name == "factors" and (requested.parent.parent / "config.frozen.json").is_file():
        root = requested.parent.parent
    else:
        root = requested
    config = load_json(root / "config.frozen.json")
    audit = load_json(root / "audit.json")
    manifest = load_json(root / "panel_manifest.json")
    result = load_json(root / "results.json")
    failure = load_json(root / "failure.json")
    issues = list(audit.get("blocking", []))
    expected_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() if config else None
    consistent = bool(expected_digest and audit.get("config_digest") == manifest.get("config_digest") == result.get("config_digest") == expected_digest)
    if config.get("schema_version") != 2:
        issues.append("冻结配置不是新版单因子回测 schema_version=2；旧交互研究产物不再展示。")
    if not audit.get("gate_passed"):
        if not issues:
            issues.append("尚未通过数据审计。")
    if manifest.get("status") != "ready":
        issues.append(f"面板未就绪：{manifest.get('status', '尚未构造')}。")
    if not result or result.get("status") != "completed" or result.get("schema_version") != 2:
        issues.append("尚无当前版本已完成的单因子回测结果。")
    if result and not consistent:
        issues.append("冻结配置、审计、面板和结果的版本指纹不一致；已有结果不作为当前结果展示。")
    if result and result.get("dataset_kind") != config.get("dataset_kind"):
        issues.append("结果数据类型与冻结配置不一致。")
    if failure:
        issues.append(str(failure.get("message", failure.get("error", "执行失败，详见 failure.json。"))))
    usable = bool(config.get("schema_version") == 2 and audit.get("gate_passed") and manifest.get("status") == "ready"
                  and result.get("schema_version") == 2 and result.get("status") == "completed" and consistent
                  and result.get("dataset_kind") == config.get("dataset_kind") and not failure)
    sections = [_synthetic(config), _proxy_notice(config), "<h1>单因子测试报告</h1>",
                "<p>分别检验 ROIC 与公募持仓变化因子的覆盖、RankIC、十分组、多空差异、超额表现及理论换手；不训练预测模型。</p>"]
    if not usable:
        sections.append(_blocked(issues))
    else:
        sections.append(_notice("当前单因子回测已完成", "逐因子报告展示冻结配置所选的信号结果。持仓直接按权重计算，不调用组合优化器；费前理论组合仅用于因子评价。", "info"))
        sections.append("<h2>基准与运行概要</h2>" + _records(result.get("benchmark", {})))
    sections.append(_quality_disclosure(config, audit, manifest if consistent else {}, usable))
    current_panel = bool(config.get("schema_version") == 2 and expected_digest
                         and audit.get("config_digest") == manifest.get("config_digest") == expected_digest
                         and manifest.get("status") == "ready")
    coverage = _csv(root / "coverage.csv") if current_panel else pd.DataFrame()
    factors: dict[str, dict] = {}
    for item in config.get("factors", []):
        if isinstance(item, dict):
            factors[str(item.get("name", ""))] = dict(item)
    if usable:
        for item in result.get("factors", []):
            if isinstance(item, dict):
                name = str(item.get("name", ""))
                factors[name] = factors.get(name, {}) | item
    completed_names = {str(item.get("name")) for item in result.get("factors", []) if isinstance(item, dict)} if usable else set()
    cards = []
    current_folders = set()
    for name, item in factors.items():
        folder = _factor_path(root, item)
        if folder is None:
            cards.append(_notice("无效因子输出路径", f"因子 {name} 的输出目录不符合 factors/因子标识 契约。"))
            continue
        current_folders.add(folder)
        factor_usable = usable and name in completed_names
        factor_issues = issues if not usable else ([] if factor_usable else ["该因子未包含在当前已完成的结果中。"])
        output = _factor_report(root, folder, item, config, audit, coverage, factor_usable, factor_issues)
        href = output.relative_to(root).as_posix()
        cards.append(f'<article class="card"><h3>{_escape(item.get("label", name))}</h3><p class="mono">{_escape(name)}</p>'
                     f'<p><a href="{_escape(href)}">打开因子完整测试报告 →</a></p>'
                     f'<p class="muted">{"费前回测产物；参阅报告中的覆盖及缺口状态" if factor_usable else "数据或结果门槛尚未通过"}</p></article>')
    # Bookmarked old reports must not continue presenting an obsolete successful run.
    factor_root = root / "factors"
    if factor_root.is_dir():
        for old in factor_root.glob("*/report.html"):
            if old.parent.resolve() not in current_folders and not old.is_symlink():
                _write(old, "过期因子报告", [_synthetic(config), "<h1>过期因子报告</h1>", _blocked(["该因子不属于当前冻结配置或已完成结果。"]), '<a href="../../report.html">返回测试汇总</a>'])
    if cards:
        sections.append('<h2>逐因子报告</h2><div class="cards">' + "".join(cards) + "</div>")
    if not coverage.empty:
        charts = _Charts()
        coverage_series = []
        if "factor" in coverage:
            for name, frame in coverage.groupby("factor", sort=False):
                coverage_series.extend(_coverage_series(frame, config, prefix=str(name) + " "))
        sections.append(charts.chart("逐因子覆盖率", coverage_series, unit="percent", caption="覆盖数据来自当前审计与面板；缺失收益不会用于重排股票或改变当期排名。"))
        sections.append(_details("逐周覆盖明细", render_table(coverage, 40)))
    sections.append(_definitions(config))
    sections = [_contextualize(section, config) for section in sections]
    sections.append(_source_audit(audit))
    for filename, title in [("preprocessing_failures", "预处理与分组失败记录"), ("field_mapping", "因子分类表映射与来源修正")]:
        if filename == "preprocessing_failures" and not current_panel:
            continue
        frame = _csv(root / f"{filename}.csv")
        if not frame.empty:
            if filename == "preprocessing_failures":
                frame = _prioritize_diagnostics(frame)
            sections.append(_details(title, render_table(frame, 80)))
    if manifest:
        sections.append(_details("冻结面板与版本追溯", _records(manifest)))
    sections.append(_artifact_links(root))
    output = _write(root / "report.html", "单因子测试报告", sections)
    if requested != root and (requested / "report.html").is_file():
        return requested / "report.html"
    return output
