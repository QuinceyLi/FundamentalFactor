"""Read-only GFOptim input reuse, without importing strategy code or credentials.

Exports are candidates, not certified historical metadata. In particular, a zero
volume proxy and a quote-derived calendar do not assert exchange provenance.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import columns, dates, fingerprint, read_frame, require_unique, write_json


INDUSTRIES = ["Agriculture", "Automobiles", "Banks", "BasicChemicals", "BuildMater", "Coal",
    "Computers", "Conglomerates", "Construction", "Defense", "FoodBeverages", "HealthCare",
    "HomeAppliances", "LightIndustry", "Machinery", "Media", "NonbankFinan", "NonferrousMetals",
    "Petroleum", "PowerUtilities", "RealEstate", "RetailTrade", "Steel", "Telecoms", "TextileGarment",
    "Transportation", "ConsumerServices", "DiverseFinan", "Electronics", "PowerEquipNewEnergy"]


def status_from_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """GF rule volume==0 or close==0; invalid inputs remain unknown, not False.

    Directional limits require actual limit prices. Historical board-name
    heuristics ignore ST, IPO exemptions and rule changes, so are not used.
    """
    result = quotes[["date", "code"]].copy()
    valid = (np.isfinite(quotes.volume) & quotes.volume.ge(0)
             & np.isfinite(quotes.close) & quotes.close.ge(0))
    result["suspended"] = (quotes.volume.eq(0) | quotes.close.eq(0)).astype("boolean").where(valid)
    result["status_method"] = "volume_or_zero_close_proxy"
    for flag, price, compare in (("limit_up", "limit_high", "ge"), ("limit_down", "limit_low", "le")):
        result[flag] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        if price in quotes:
            known = np.isfinite(quotes[price]) & quotes[price].gt(0) & np.isfinite(quotes.open) & quotes.open.gt(0)
            result.loc[known, flag] = getattr(quotes.loc[known, "open"], compare)(quotes.loc[known, price])
    return result


def combine_raw_quotes(old: pd.DataFrame, recent: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prefer the newer raw feed per key; never splice adjusted prices/factors."""
    for frame in (old, recent):
        require_unique(frame, ["date", "code"])
    common = old.merge(recent, on=["date", "code"], suffixes=("_old", "_new"))
    diagnostic = {"overlap_rows": len(common)}
    for field in ("open", "close", "volume"):
        diagnostic[f"{field}_disagreements"] = int((~np.isclose(common[f"{field}_old"],
            common[f"{field}_new"], rtol=1e-6, atol=1e-6, equal_nan=True)).sum())
    first, second = old.copy(), recent.copy()
    first["quote_source"], second["quote_source"] = "legacy_raw", "recent_raw"
    # Do not rely on caller's input order to resolve duplicates within either file.
    result = pd.concat([first, second], ignore_index=True).drop_duplicates(["date", "code"], keep="last")
    return result.sort_values(["date", "code"]).reset_index(drop=True), diagnostic


def industry_candidates(exposures: pd.DataFrame) -> pd.DataFrame:
    missing = set(INDUSTRIES) - set(exposures)
    if missing:
        raise ValueError(f"缺少行业哑变量：{sorted(missing)}")
    values = exposures[INDUSTRIES]
    good = values.isin([0, 1]).all(axis=1) & values.sum(axis=1).eq(1)
    result = exposures[["date", "code", "secShortName"]].copy()
    result["industry"] = values.idxmax(axis=1).where(good)
    result["industry_onehot_valid"] = good
    result["st_name_hint"] = exposures.secShortName.astype("string").str.upper().str.contains("ST", regex=False)
    # Hints must not be accidentally accepted as the canonical is_st/float_mcap history.
    return result


def prepare_inputs(output: Path, old_path: Path, recent_path: Path, exposure_path: Path,
                   actions_path: Path, start: str = "2013-01-01") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    paths = {"old_quotes": old_path, "recent_quotes": recent_path,
             "industry": exposure_path, "corporate_actions": actions_path}
    if any(output.resolve() == p.resolve() or output.resolve() in p.resolve().parents for p in paths.values()):
        raise ValueError("输出目录不得包含源文件")
    read, quarantined = [], []
    for path in (old_path, recent_path):
        available = columns(path)
        fields = ["date", "code", "open", "close", "volume"] + [c for c in ("last", "limit_high", "limit_low") if c in available]
        # The legacy file also contains T00018.SH. Quarantine noncanonical
        # instruments explicitly instead of silently treating them as A shares.
        if path.suffix == ".feather":
            frame = pd.read_feather(path, columns=fields)
        elif path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=fields)
        else:
            frame = pd.read_csv(path, usecols=fields, dtype={"code": "string"})
        frame["date"] = dates(frame.date)
        frame["code"] = frame.code.astype("string").str.strip().str.upper()
        canonical = frame.code.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)", na=False)
        rejected = frame.loc[~canonical, ["date", "code"]].copy()
        rejected["source"] = str(path)
        quarantined.append(rejected)
        frame = frame[canonical]
        read.append(frame[frame.date.ge(pd.Timestamp(start))].copy())
    quotes, overlap = combine_raw_quotes(*read)
    del read
    excluded = pd.concat(quarantined, ignore_index=True)
    excluded.to_csv(output / "noncanonical_quote_keys.csv", index=False)
    quotes.to_parquet(output / "raw_quotes.parquet", index=False)
    state = status_from_quotes(quotes)
    state.to_parquet(output / "status_candidates.parquet", index=False)
    # Quote-derived calendar includes older sessions needed to calculate listing age.
    all_days = pd.concat([read_frame(path, ["date"]).date for path in (old_path, recent_path)]).drop_duplicates().sort_values()
    all_days.to_frame("date").to_parquet(output / "calendar_candidates.parquet", index=False)
    exp = read_frame(exposure_path, ["date", "code", "secShortName"] + INDUSTRIES)
    require_unique(exp, ["date", "code"])
    industry = industry_candidates(exp)
    industry.to_parquet(output / "industry_candidates.parquet", index=False)
    actions = read_frame(actions_path, ["date", "code", "share_ratio", "cash_per_share", "notice_date"])
    require_unique(actions, ["date", "code"])
    actions.to_parquet(output / "corporate_actions_partial.parquet", index=False)
    generated = ["raw_quotes.parquet", "status_candidates.parquet", "calendar_candidates.parquet",
                 "industry_candidates.parquet", "corporate_actions_partial.parquet", "noncanonical_quote_keys.csv"]
    result = {"formal_ready": False, "source_fingerprints": {name: fingerprint(path) for name, path in paths.items()},
        "artifacts": {name: fingerprint(output / name) for name in generated}, "raw_overlap": overlap,
        "quarantined_noncanonical": {"rows": len(excluded), "codes": excluded.code.drop_duplicates().tolist()},
        "quotes": {"rows": len(quotes), "start": quotes.date.min(), "end": quotes.date.max(),
                   "codes": quotes.code.nunique(), "bj_last_date": quotes.loc[quotes.code.str.endswith(".BJ"), "date"].max()},
        "status": {"suspended_proxy_rows": int(state.suspended.sum()),
                   "unknown_suspension_rows": int(state.suspended.isna().sum()),
                   "unknown_limit_up_rows": int(state.limit_up.isna().sum()),
                   "unknown_limit_down_rows": int(state.limit_down.isna().sum()),
                   "method": "GFOptim volume==0 or close==0；缺失状态未知；未提供正式涨跌停价格不作板块比例推断"},
        "industry": {"rows": len(industry), "valid_onehot_fraction": float(industry.industry_onehot_valid.mean()),
                     "start": industry.date.min(), "end": industry.date.max(), "is_st_verified": False},
        "contracts": {"calendar_verified": False, "status_verified": False, "corporate_actions_complete": False,
                      "adjusted_price_repair": "none; only raw prices exported"}}
    write_json(output / "adapter_manifest.json", result)
    write_data_report(output)
    return result


def write_data_report(output: Path) -> Path:
    inventory = output / "reference_inventory.json"
    details = json.loads(inventory.read_text(encoding="utf-8")) if inventory.exists() else {}
    adapter = output / "adapter_manifest.json"
    manifest = json.loads(adapter.read_text(encoding="utf-8")) if adapter.exists() else {}
    gaps = [
        ("历史证券属性", "上市/退市日期、历史A股资格及代码变更；不能用首次行情日期代替上市日"),
        ("ST及历史行业", "正式ST/*ST历史；2013—2023行业及北交所行业。2024年以来行业和股票名称已有核验线索"),
        ("实际流通市值", "逐日或按有效日更新的实际流通市值；SIZE、加工tmv、指数调整股本均不能替代"),
        ("交易日历与涨跌停", "需确认行情推导日历完整性，补历史官方涨跌停价/无涨跌幅限制区间；停牌已有成交量推导输入"),
        ("公司行动与价格", "修复已确认个股复权异常并核实全收益口径；执行账本另需完整分红送转/配股合并处理，现有事件仅覆盖近段且全A未闭合"),
        ("退市", "理论标签需要同一全收益尺度的终止值；执行账本需要实际可知结算日及每当前股现金"),
        ("北交所尾段", "2026-07-09—2026-08-20原始行情/状态；新行情仅沪深，不能补足全A尾段"),
    ]
    rows = "".join(f"<tr><th>{escape(a)}</th><td>{escape(b)}</td></tr>" for a, b in gaps)
    quote_info, industry_info = manifest.get("quotes", {}), manifest.get("industry", {})
    ready = [
        ("未复权行情", f"{quote_info.get('rows', '待导出')} 行；{quote_info.get('start', '')} 至 {quote_info.get('end', '')}", "新旧逐键合并；不含复权因子修复"),
        ("停牌输入", f"{manifest.get('status', {}).get('suspended_proxy_rows', '待导出')} 条零成交量/零收盘代理记录", "不是正式开盘前状态；未知涨跌停不放行"),
        ("行业", f"{industry_info.get('rows', '待导出')} 行；{industry_info.get('start', '')} 至 {industry_info.get('end', '')}", "30行业哑变量可转换；没有实际流通市值，名称不能代替历史ST"),
        ("公司行动", "现有CSV 8,753事件，主要为2025—2026", "原审计另有37条未匹配记录，全A完整性未确认"),
    ]
    ready_rows = "".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in ready)
    html = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GFOptim 数据复用与缺项核查</title>
<style>body{{font:16px/1.7 sans-serif;max-width:1150px;margin:40px auto;padding:0 20px;color:#172534}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dc;padding:12px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:16px}}.alert{{background:#fff0dd;padding:16px}}</style>
<h1>GFOptim 数据复用与缺项核查</h1><p class="alert">当前未通过正式回测数据门槛。此页报告输入事实及代码能力，不包含真实因子收益结论。</p>
<h2>已找到并适配的数据</h2><table><tr><th>数据</th><th>覆盖</th><th>限制</th></tr>{ready_rows}</table>
<h2>已经接入的处理</h2><p>停牌双向禁止、涨停禁止买入、跌停禁止卖出；无法卖出的旧仓持续估值；先卖后买，资金不足按比例缩减买单。每周开盘调仓，不每日重试。分红和送转按显式事件修改现金和股数；缺报价时保持经过公司行动调整的上次估值，绝不用延续估值成交。未知状态冻结交易并留痕。该账本使用分数股和零费税，仅是成交限制敏感性分析。</p>
<h2>已确认价格异常</h2><p>000001.SZ 在 2025-05-27 → 2025-05-28 原始收盘 11.49 → 11.53（约 +0.348%），旧行情及 /data/mongo/quotefac.feather 的复权收盘却为 1468.2433 → 904.4114（约 −38.4%）。新行情两日复权因子均为 78.439843；旧行情因子从 127.784447 跌至 78.439843。已确认的是这一个股案例，尚未认定全部数据均存在该问题。本次适配器不混接复权因子、不自动修复源数据。</p>
<h2>仍需补齐或确认</h2><table>{rows}</table>
<h2>复用产物与限制</h2><p>raw_quotes.parquet 为逐键优先使用近期源的未复权行情；status_candidates.parquet 为成交量推导状态，未知涨跌停保持空值；calendar_candidates.parquet 为行情日期并集；industry_candidates.parquet 只保存唯一有效行业及股票名称ST线索；corporate_actions_partial.parquet 不声明全历史或全A完整。候选文件不能直接替代历史证券主表、已确认行业市值/ST接口或完整事件源。</p>
<details open><summary>适配器审计与版本指纹</summary><pre>{escape(json.dumps(manifest, ensure_ascii=False, indent=2))}</pre></details>
<details><summary>只读源文件核查明细</summary><pre>{escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></details></html>'''
    target = output / "data_requirements.html"
    target.write_text(html, encoding="utf-8")
    return target
