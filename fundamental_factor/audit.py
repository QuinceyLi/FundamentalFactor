from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, RISK_FIELDS
from .data import columns, fingerprint, read_frame, require_unique, write_json


def optional_columns(path: Path | None) -> list[str]:
    """Probe optional fields; the audit loop reports unreadable files normally."""
    if path is None or not path.is_file():
        return []
    try:
        return columns(path)
    except (OSError, ValueError, TypeError):
        return []


def source_columns(config: Config) -> dict[str, list[str]]:
    price = config.raw["price_contract"]
    classic = config.data_policy["mode"] == "classic"
    history = ["date", "code"]
    if config.requires_industry:
        history.append("industry")
    if config.requires_float_mcap:
        history.append("float_mcap")
    path = config.source("history")
    if not classic or "is_st" in optional_columns(path):
        history.append("is_st")
    status = ["date", "code"]
    if not classic or "suspended" in optional_columns(config.source("status")):
        status.append("suspended")
    result = {
        "quotes": ["date", "code", price.get("open_column", "acc_open"), price.get("close_column", "acc_close")],
        "stocks": ["code", "listing_date", "delisting_date"],
        "history": history,
        "calendar": ["date"], "status": status,
        "terminal": ["date", "available_date", "code", "terminal_value"],
    }
    if config.requires_risk:
        result["risk"] = ["date", "code"] + RISK_FIELDS
    if config.raw.get("execution", {}).get("enabled"):
        result["execution_status"] = ["date", "code", "suspended", "limit_up", "limit_down"]
        if config.raw["execution"].get("price_mode", "raw") == "raw":
            result.update({
                "execution_quotes": ["date", "code", "open", "close"],
                "corporate_actions": ["date", "code", "share_ratio", "cash_per_share", "notice_date"],
                "execution_terminal": ["date", "code", "cash_per_share", "available_date"],
            })
    for factor in config.factors:
        name = factor["source"]
        if name in {"quotes", "risk", "stocks", "history", "calendar", "status", "terminal",
                    "execution_quotes", "execution_status", "corporate_actions", "execution_terminal"}:
            raise ValueError("因子来源键不能使用 quotes/risk/stocks/history/calendar/status/terminal 保留名称")
        result.setdefault(name, ["date", "code"])
        if factor["column"] not in result[name]:
            result[name].append(factor["column"])
    return result


def mapping_audit(config: Config) -> tuple[list[dict], list[str]]:
    mapping, warnings = [], []
    workbook = config.raw.get("classification_workbook")
    if not workbook:
        return [], []
    try:
        from openpyxl import load_workbook
        book = load_workbook(workbook, read_only=True, data_only=True)
        for row in list(book.active.values)[1:]:
            if len(row) < 4 or not row[0]:
                continue
            name, level1, level2, source, *_ = row
            resolved = "alternative_incentive_amount.feather" if name == "incentive_amount" else source
            item = {"factor": name, "level1": level1, "level2": level2, "workbook_source": source,
                    "resolved_source": resolved, "correction": name == "incentive_amount"}
            if resolved:
                path = Path(config.raw["data_root"]) / resolved
                item["column_present"] = path.is_file() and name in columns(path)
                if not item["column_present"]:
                    warnings.append(f"分类表字段不可用：{name} / {resolved}")
            mapping.append(item)
        book.close()
    except (OSError, ValueError, KeyError) as exc:
        warnings.append(f"分类表读取失败：{exc}")
    return mapping, warnings


def audit(config: Config) -> dict:
    config.output.mkdir(parents=True, exist_ok=True)
    frozen = config.output / "config.frozen.json"
    if frozen.exists():
        import json
        previous = json.loads(frozen.read_text(encoding="utf-8"))
        if previous != config.raw:
            from .config import DataGateError
            raise DataGateError("输出目录已有不同冻结配置；请使用新的 output_dir")
    result = {"schema_version": 2, "config_digest": config.digest, "gate_passed": False,
              "blocking": [], "warnings": [], "sources": {},
              "execution_gate_passed": True, "execution_blocking": [],
              "point_in_time": config.raw.get("point_in_time_evidence", ""),
              "data_policy": config.data_policy}
    classic = config.data_policy["mode"] == "classic"
    relaxed = config.data_policy["mode"] != "strict"
    execution_names = {"execution_quotes", "execution_status", "corporate_actions", "execution_terminal"}
    optional = {"terminal", "status"} if classic else {"terminal", "execution_terminal"}
    optional.add("execution_terminal")
    if classic and not config.requires_industry and not config.requires_float_mcap:
        optional.add("history")

    def blocking(name: str, message: str) -> None:
        result["execution_blocking" if name in execution_names else "blocking"].append(message)

    sets, quote_dates, stock_codes, earliest_listing = {}, None, None, None
    requirements = source_columns(config)
    for name, required in requirements.items():
        path = config.source(name)
        if path is None or not path.is_file():
            if name not in optional or path is not None and not classic:
                blocking(name, f"缺少 {name} 数据入口或文件")
            elif classic:
                result["warnings"].append(f"可选 {name} 数据未提供；保留未知状态或缺失收益披露，不阻断因子统计。")
            continue
        try:
            available = columns(path)
            if missing := set(required) - set(available):
                raise ValueError(f"missing columns {sorted(missing)}")
            selected = required + (["is_a_share"] if name == "stocks" and "is_a_share" in available else [])
            frame = read_frame(path, selected, nullable_booleans=(name == "execution_status" or
                                relaxed and name in ("history", "status")))
            require_unique(frame, [c for c in ("date", "code") if c in frame])
            empty_optional = classic and frame.empty and name in optional
            if frame.empty and name not in ("corporate_actions", "execution_terminal") and not empty_optional:
                raise ValueError("empty source")
            info = {"fingerprint": fingerprint(path), "rows": len(frame), "available_columns": available,
                    "selected_columns": selected, "duplicate_keys": 0, "numeric": {}}
            if "date" in frame:
                info.update(start=frame.date.min(), end=frame.date.max(), dates=frame.date.nunique())
            if "code" in frame:
                sets[name] = set(frame.code)
                info["codes"] = len(sets[name])
            if empty_optional:
                info["empty_optional"] = True
                if name in ("history", "status"):
                    info["unknown_flags"] = {"is_st" if name == "history" else "suspended": 0}
                result["sources"][name] = info
                result["warnings"].append(f"可选 {name} 数据为空；无状态或结算证据，保留未知状态或缺失收益披露。")
                continue
            for column in frame.select_dtypes(include="number"):
                values = frame[column].to_numpy(dtype=float)
                finite = np.isfinite(values)
                info["numeric"][column] = {"finite": int(finite.sum()), "missing_or_nonfinite": int((~finite).sum()),
                                           "zero": int((values == 0).sum()),
                                           "min": float(values[finite].min()) if finite.any() else None,
                                           "max": float(values[finite].max()) if finite.any() else None}
            if name == "quotes":
                quote_dates = pd.DatetimeIndex(frame.date.unique()).sort_values()
            if name == "stocks":
                stock_codes = sets[name]
                earliest_listing = frame.listing_date.min()
                if (frame.delisting_date.notna() & frame.delisting_date.lt(frame.listing_date)).any():
                    raise ValueError("delisting_date precedes listing_date")
                if "is_a_share" not in available:
                    result["warnings"].append("stocks 未提供 is_a_share；接口约定该文件仅包含历史A股，需由数据提供方保证。")
            if name == "status" and not {"limit_up", "limit_down"}.issubset(available):
                result["warnings"].append("涨跌停标记不完整；交易限制披露不覆盖缺失类别。")
            if relaxed and name in ("history", "status"):
                flag = "is_st" if name == "history" else "suspended"
                info["unknown_flags"] = {flag: int(frame[flag].isna().sum()) if flag in frame else len(frame)}
                if flag not in frame:
                    result["warnings"].append(f"{name} 缺少 {flag}；状态按未知处理，未声称完整历史覆盖。")
            if name == "calendar" and quote_dates is not None:
                if len(quote_dates.difference(frame.date)):
                    raise ValueError("calendar omits quote trading dates")
            if name == "terminal":
                if not np.isfinite(frame.terminal_value).all() or frame.terminal_value.lt(0).any():
                    raise ValueError("terminal_value must be a nonnegative total-return level")
                if frame.available_date.lt(frame.date).any():
                    raise ValueError("terminal available_date precedes effective date")
            if name == "corporate_actions":
                values = frame[["share_ratio", "cash_per_share"]].to_numpy(dtype=float)
                if not np.isfinite(values).all() or (values < 0).any() or frame.notice_date.gt(frame.date).any():
                    raise ValueError("公司行动必须非负且在生效日已公布；配股/合并等其他行动需另行核实")
            if name == "execution_terminal":
                require_unique(frame, ["code"])
                if not np.isfinite(frame.cash_per_share).all() or frame.cash_per_share.lt(0).any() or frame.available_date.gt(frame.date).any():
                    raise ValueError("执行终止事件须按实际可知结算日入账，现金每当前股非负")
            if name == "execution_status":
                info["unknown_flags"] = frame[["suspended", "limit_up", "limit_down"]].isna().sum().to_dict()
                if sum(info["unknown_flags"].values()):
                    result["warnings"].append("执行状态含未知值；对应方向禁止成交，继续持仓估值。")
            result["sources"][name] = info
            del frame
        except (OSError, ValueError, KeyError, TypeError) as exc:
            blocking(name, f"{name} 接入校验失败：{exc}")
    if stock_codes is not None:
        primary = set().union(*(sets.get(f["source"], set()) for f in config.factors), sets.get("quotes", set()))
        if config.exchanges is not None:
            primary = {code for code in primary if code.rsplit(".", 1)[-1] in config.exchanges}
        missing = sorted(primary - stock_codes)
        if missing:
            result["blocking"].append(f"历史股票属性未覆盖 {len(missing)} 个代码，例如 {missing[:5]}")
    calendar_path = config.source("calendar")
    if earliest_listing is not None and quote_dates is not None and "calendar" in result["sources"]:
        calendar = read_frame(calendar_path, ["date"])
        if calendar.date.min() > earliest_listing and (calendar.date < quote_dates.min()).sum() < config.raw["backtest"]["min_listing_sessions"]:
            result["blocking"].append("交易日历缺少计算上市年龄所需的样本前历史")
    if not config.raw.get("point_in_time_confirmed") or not config.raw.get("point_in_time_evidence"):
        result["warnings" if classic else "blocking"].append("未确认因子及历史控制变量的信息可用时点；仍严格向后取快照，但未独立证明源数据当时可用。")
    contract = config.raw["price_contract"]
    adjusted_proxy = relaxed and contract.get("return_kind") == "adjusted_price"
    if not (contract.get("verified") and (contract.get("total_return") or adjusted_proxy)
            and contract.get("description", "").strip()):
        result["warnings" if classic else "blocking"].append("缺少已验证的全收益价格口径说明（复权、分红、公司行动）；结果仅反映输入价格变化，不自动等于完整现金全收益。")
    if relaxed:
        if not classic and not config.data_policy["description"].strip():
            result["blocking"].append("代理研究必须说明 ST、停牌及涨跌停代理的来源和局限")
        result["warnings"].append("已知ST优先剔除；未知ST按冻结配置处理，未知不等于非ST。")
        result["warnings"].append("入场日全天成交量停牌代理属于事后可见的研究过滤口径，不能视为开盘前已知的可成交状态；不使用退出日状态筛选信号。")
        if adjusted_proxy:
            result["warnings"].append("收益按核验后的复权价格变化计算，未将完整公司行动或实际现金结算覆盖视为已验证的全收益。")
        if contract.get("suspension_valuation", "previous_adjusted_close" if classic else None) == "previous_adjusted_close":
            result["warnings"].append("停牌端点延续此前复权收盘价估值；这是简化估值，非该日实际成交价或完整现金分红记账。")
        if config.data_policy["missing_returns"] == "report":
            result["warnings"].append("冻结信号后缺失收益不剔除持仓、不补零；IC按可用标签计算并披露覆盖率，不完整组合收益及连续绩效记为缺失。")
        elif config.data_policy["missing_returns"] == "available_case":
            result["warnings"].append("有效收益样本统计：先冻结因子分组，再对有收益标签的股票归一化统计权重；基准同口径。原始缺失标签不补零，报告覆盖率和缺失目标权重；该统计组合可能有缺失样本选择偏差，不等同完整目标持仓的实际收益。")
        if config.data_policy["empty_groups"] == "cash":
            result["warnings"].append("当周整体信号存在但某十分组因并列为空时，按100%现金、零利息处理；整体信号缺失或非空组收益全部缺失不视为空仓。")
    execution = config.raw.get("execution", {})
    if execution.get("enabled"):
        adjusted_units = execution.get("price_mode", "raw") == "adjusted_units"
        if adjusted_units:
            if any(config.source(name) is not None for name in ("corporate_actions", "execution_terminal")):
                result["execution_blocking"].append("复权单位账本禁止接入原始股数口径公司行动或现金终止结算，以免重复调整或单位混用")
            result["warnings"].append("复权单位账本按复权价格和代理状态模拟方向限制与现金约束；单位数不是实际股数，未重复入账分红送转；持有退市未解决时对应账户停止报告收益。")
        else:
            for key, explanation in (("raw_prices_verified", "未复权价格及币种单位"),
                                     ("status_verified", "历史停牌与开盘涨跌停状态"),
                                     ("corporate_actions_complete", "全样本公司行动完整性（含无事件代码）")):
                if execution.get(key) is not True:
                    result["execution_blocking"].append(f"执行账本尚未确认：{explanation}")
        if not execution.get("description", "").strip():
            result["execution_blocking"].append("执行账本缺少价格、状态及公司行动口径说明")
        if not adjusted_units or "coverage_start" in execution or "coverage_end" in execution:
            try:
                coverage_start = pd.Timestamp(execution["coverage_start"])
                coverage_end = pd.Timestamp(execution["coverage_end"])
                if pd.isna(coverage_start) or pd.isna(coverage_end) or coverage_start > coverage_end:
                    raise ValueError("invalid coverage")
            except (KeyError, TypeError, ValueError):
                result["execution_blocking"].append("执行账本需要声明已核实的 coverage_start / coverage_end")
        result["execution"] = execution
    result["execution_gate_passed"] = not result["execution_blocking"]
    if result["execution_blocking"]:
        if classic:
            result["warnings"].append("可选成交账本资料不足，跳过成交模拟；因子IC及理论权重组合继续计算。")
            result["warnings"].extend(result["execution_blocking"])
        else:
            result["blocking"].extend(result["execution_blocking"])
    result["warnings"].append("历史公开可用时点依赖已确认的数据生产口径；快照抽查不等于独立公告审计。")
    mapping, warnings = mapping_audit(config)
    result["mapping"] = mapping
    result["warnings"].extend(warnings)
    result["gate_passed"] = not result["blocking"]
    write_json(frozen, config.raw)
    write_json(config.output / "audit.json", result)
    write_json(config.output / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "pyarrow", "scipy", "statsmodels", "openpyxl")}})
    pd.DataFrame(mapping, columns=["factor", "level1", "level2", "workbook_source", "resolved_source",
                                  "correction", "column_present"]).to_csv(config.output / "field_mapping.csv", index=False)
    return result
