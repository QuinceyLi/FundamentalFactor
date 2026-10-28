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


def market_cap_source_columns(config: Config) -> dict[str, list[str]]:
    """Proxy inputs share normal source keys, including the core quotes source."""
    if not config.uses_market_cap_proxy:
        return {}
    spec = config.market_cap
    result = {spec["quotes_source"]: [spec["close_column"], spec["volume_column"]]}
    result.setdefault(spec["turnover_source"], [])
    if spec["turnover_column"] not in result[spec["turnover_source"]]:
        result[spec["turnover_source"]].append(spec["turnover_column"])
    return result


def source_columns(config: Config) -> dict[str, list[str]]:
    price = config.raw["price_contract"]
    classic = config.data_policy["mode"] == "classic"
    history = ["date", "code"]
    history_available = optional_columns(config.source("history"))
    industry_column = config.industry["column"]
    if (config.requires_industry and not config.separate_industry_source
            and (not config.optional_control_sources or industry_column in history_available)):
        history.append(industry_column)
    if (config.requires_float_mcap and not config.separate_market_cap_source
            and (not config.optional_control_sources or "float_mcap" in history_available)):
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
        exposure_columns = RISK_FIELDS
        if config.optional_control_sources:
            available = optional_columns(config.source("risk"))
            exposure_columns = [column for column in RISK_FIELDS if column in available]
        result["risk"] = ["date", "code"] + exposure_columns
    if config.separate_industry_source:
        name = config.industry["source"]
        result.setdefault(name, ["date", "code"])
        if not config.optional_control_sources or industry_column in optional_columns(config.source(name)):
            if industry_column not in result[name]:
                result[name].append(industry_column)
    if config.separate_market_cap_source:
        name, field = config.market_cap["source"], config.market_cap["column"]
        result.setdefault(name, ["date", "code"])
        if not config.optional_control_sources or field in optional_columns(config.source(name)):
            if field not in result[name]:
                result[name].append(field)
    if config.raw.get("execution", {}).get("enabled"):
        result["execution_status"] = ["date", "code", "suspended", "limit_up", "limit_down"]
        if config.raw["execution"].get("price_mode", "raw") == "raw":
            result.update({
                "execution_quotes": ["date", "code", "open", "close"],
                "corporate_actions": ["date", "code", "share_ratio", "cash_per_share", "notice_date"],
                "execution_terminal": ["date", "code", "cash_per_share", "available_date"],
            })
    for name, fields in market_cap_source_columns(config).items():
        available = optional_columns(config.source(name)) if config.optional_control_sources else fields
        result.setdefault(name, ["date", "code"])
        result[name].extend(field for field in fields if field in available and field not in result[name])
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
    if (classic and (not config.requires_industry or config.separate_industry_source)
            and (not config.requires_float_mcap or config.separate_market_cap_source)):
        optional.add("history")
    if config.optional_control_sources:
        optional.update({"history", "risk"})
        required_names = {"quotes", "stocks", "calendar"} | {factor["source"] for factor in config.factors}
        if config.raw.get("execution", {}).get("enabled"):
            required_names.update(execution_names)
        optional.update(set(market_cap_source_columns(config)) - required_names)
        if config.separate_industry_source and config.industry["source"] not in required_names:
            optional.add(config.industry["source"])
        if config.separate_market_cap_source and config.market_cap["source"] not in required_names:
            optional.add(config.market_cap["source"])
        result["warnings"].append("各信号版本独立取样：原始版本不要求行业、市值或Barra；控制数据缺失仅影响需要它们的中性化版本。版本间表现差异也可能来自样本差异。")
    proxy_fields = market_cap_source_columns(config)
    result["smoothing"] = config.smoothing
    result["industry"] = config.industry
    result["missing_values"] = config.missing_values
    if config.aiq_fill_enabled:
        result["warnings"].append("缺失处理采用AIQ字段矩阵规则：volume/amount/return/vwap首个非空后补0；open/high/low/turnover等保留；acc_open/acc_close、市值、普通因子等前填；日内指定字段用last。保留原值、原始缺失与真实供值日；不会向未来回填，ST仍按事件而非矩阵填充。")
    if config.industry["classification"] == "citic_l1":
        result["warnings"].append("行业采用中信一级规范名称；仅使用截点前已存在的历史记录，未知或非中信原始类别保持缺失，不用当前分类回填历史。")
        from .industry import industry_catalog
        industry_catalog().to_csv(config.output / "industry_mapping.csv", index=False)
    elif config.industry["classification"] == "aiq_zx35":
        from .aiq_industry import industry_catalog
        industry_catalog().to_csv(config.output / "industry_mapping.csv", index=False)
        result["warnings"].append("行业按 AIQ FACFUNC.get_industry → Toolbox.zxclassified/get_classname 的35类规则及名称处理；它由中信多级行业派生，不等同中信一级30类。保留原始层级和有效区间，空缺/冲突不向过去填充。")
    if config.uses_market_cap_proxy:
        result["market_cap"] = {**config.market_cap, "column": config.market_cap_column,
                                "formula": "close * volume * 100 / (turnover / 100)",
                                "is_proxy": True}
        result["warnings"].append("市值中性化使用未复权收盘价 × 成交量(手) × 100 ÷ (换手率(%) / 100) 的行情代理；不声称等同已验证的实际流通市值。仅配对同日且截点前的原始价格/成交量/换手率，非正或非有限输入按缺失处理。")
    elif config.requires_float_mcap:
        result["market_cap"] = {**config.market_cap, "column": "float_mcap", "is_proxy": False}
        if config.separate_market_cap_source:
            result["warnings"].append("流通市值从独立历史来源按截点向后匹配；float_mcap 单位为元。" +
                ("AIQ模式仅对NaN前填，保留原始0/负/无穷不跳过；随后检查有限正值，原始无效与填充均披露。" if config.aiq_fill_enabled else
                 "显式无效更新保留缺失，不回跳到旧有效值；") + "缺失只影响需要市值控制的版本。")
    if config.raw.get("input_manifest"):
        import json
        provenance = Path(config.raw["input_manifest"])
        if provenance.is_file():
            result["input_provenance"] = {"fingerprint": fingerprint(provenance),
                                          "manifest": json.loads(provenance.read_text())}
        else:
            result["warnings"].append("接入数据的来源说明文件缺失；原始输入审计仍独立进行。")

    def blocking(name: str, message: str) -> None:
        result["execution_blocking" if name in execution_names else "blocking"].append(message)

    sets, quote_dates, stock_codes, earliest_listing = {}, None, None, None
    primary_sets = {}
    factor_sources = {factor["source"] for factor in config.factors}
    configured_end = config.raw["backtest"].get("end_date")
    attribute_check_end = pd.Timestamp(configured_end) if configured_end else None
    requirements = source_columns(config)
    for name, required in requirements.items():
        path = config.source(name)
        if path is None or not path.is_file():
            if name not in optional or path is not None and not classic:
                blocking(name, f"缺少 {name} 数据入口或文件")
            elif classic:
                result["warnings"].append(f"可选 {name} 数据未提供；保留未知状态或缺失收益披露，不阻断因子统计。")
                if config.optional_control_sources and name in ("history", "risk"):
                    result["warnings"].append(f"{name} 控制数据不可用：需要这些字段的中性化版本将记录无有效样本；原始版本继续。")
                if config.optional_control_sources and name in proxy_fields:
                    result["warnings"].append(f"{name} 市值代理来源不可用：需要市值的版本无有效代理值；原始版本继续。")
            continue
        try:
            available = columns(path)
            if missing := set(required) - set(available):
                raise ValueError(f"missing columns {sorted(missing)}")
            if config.optional_control_sources and name in ("history", "risk"):
                expected = (RISK_FIELDS if name == "risk" else
                            ([config.industry["column"]] if config.requires_industry and not config.separate_industry_source else []) +
                            (["float_mcap"] if config.requires_float_mcap and not config.separate_market_cap_source else []))
                if absent := set(expected) - set(available):
                    result["warnings"].append(f"{name} 缺少控制字段 {sorted(absent)}；仅影响需要这些字段的版本，原始版本不因此删除股票。")
            if config.optional_control_sources and name in proxy_fields:
                if absent := set(proxy_fields[name]) - set(available):
                    result["warnings"].append(f"{name} 缺少市值代理字段 {sorted(absent)}；仅影响需要市值的版本，原始版本不因此删除股票。")
            if (config.optional_control_sources and config.separate_industry_source
                    and name == config.industry["source"] and config.industry["column"] not in available):
                result["warnings"].append(f"{name} 缺少中信行业字段；行业中性化版本无有效样本，原始版本继续。")
            if (config.optional_control_sources and config.separate_market_cap_source
                    and name == config.market_cap["source"] and config.market_cap["column"] not in available):
                result["warnings"].append(f"{name} 缺少流通市值字段；仅影响需要市值的版本，原始版本继续。")
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
                price_end = quote_dates.max()
                attribute_check_end = (min(attribute_check_end, price_end)
                                       if attribute_check_end is not None else price_end)
                result["stock_attribute_check_end"] = attribute_check_end
            if name == "quotes" or name in factor_sources:
                # Old snapshots may remain valid at entry, but records arriving
                # after the last possible holding endpoint cannot enter this run.
                # A recently refreshed factor file must not make an older price
                # history require attributes for future-only securities.
                relevant = frame if attribute_check_end is None else frame[frame.date.le(attribute_check_end)]
                primary_sets[name] = set(relevant.code)
                info["codes_through_holding_end"] = len(primary_sets[name])
                del relevant
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
        primary = set().union(*(primary_sets.get(f["source"], set()) for f in config.factors), primary_sets.get("quotes", set()))
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
        if config.aiq_fill_enabled:
            result["warnings"].append("收益端点按AIQ源字段填充：acc_open延续自身历史开盘，而非前收盘。保留原始缺口和供值日；上市前供值/退市后无观察价格不作为有效交易或结算。日return补0不会直接套到周/月收益。")
        elif contract.get("suspension_valuation", "previous_adjusted_close" if classic else None) == "previous_adjusted_close":
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
