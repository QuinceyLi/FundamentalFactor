"""Validate AIQ references and prepare independent research inputs, without updating AIQ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyarrow.feather as feather

from fundamental_factor.aiq_inputs import (merge_st_sources, prepare_calendar, prepare_stocks,
                                           prepare_trade_status)
from fundamental_factor.aiq_industry import prepare_aiq_industry_history
from fundamental_factor.aiq_market_cap import prepare_aiq_market_cap
from fundamental_factor.aiq_st import prepare_aiq_st
from fundamental_factor.data import dates, fingerprint, read_frame, write_json


def prepare(root: Path, aiq: Path, mongo: Path, *, reuse_adapters=False) -> dict:
    root, aiq, mongo = root.resolve(), aiq.resolve(), mongo.resolve()
    if root.is_relative_to(aiq) or root.is_relative_to(mongo):
        raise ValueError("Prepared outputs must be outside AIQ and Mongo source directories")
    root.mkdir(parents=True, exist_ok=True)
    database = aiq / "database"
    repaired = mongo / "quotefac_repaired.feather"
    protected_price = fingerprint(repaired)
    calendar_path, stocks_path = root / "trade_calendar.parquet", root / "stock_master.parquet"
    summary = {"schema_version": 1, "aiq_root": str(aiq), "prepared_directory": str(root),
               "price_source_protected": protected_price, "aiq_modules_executed": False,
               "source_files_modified": False, "datasets": {}}
    summary["datasets"]["calendar"] = prepare_calendar(database/"presaved/tradeday.csv", mongo/"public_trade_calendar_sina.feather", calendar_path)
    summary["datasets"]["stocks"] = prepare_stocks(database/"presaved/all_stocks_code.csv", database/"presaved/delist_stock.csv", mongo/"public_stock_master_sh_sz.feather", stocks_path)
    print("Prepared calendar and security lifecycle inputs", flush=True)
    industry = root/"industry/stock_industry.parquet"
    st = root/"st/aiq_st.parquet"
    cap_dir = root/"market_cap"
    for path in (industry, st): path.parent.mkdir(parents=True, exist_ok=True)
    if not reuse_adapters or not industry.is_file():
        prepare_aiq_industry_history(database/"sector_info", industry)
    if not reuse_adapters or not st.is_file():
        prepare_aiq_st(database/"presaved/st.csv", database/"presaved/previous_name.xlsx", st,
                       calendar_source=calendar_path)
    if not reuse_adapters or not (cap_dir/"float_mcap.parquet").is_file():
        prepare_aiq_market_cap(database/"market_feature/market_feature.feather", calendar_path, cap_dir,
            share_source=database/"presaved/share.csv",
            quote_source=Path(__file__).resolve().parents[1]/"outputs/gfoptim_inputs/raw_quotes.parquet",
            proxy_turnover_source=repaired)
    for name, path in (("industry", industry.with_suffix(".manifest.json")),
                       ("st_aiq", st.with_suffix(".manifest.json")), ("market_cap", cap_dir/"manifest.json")):
        summary["datasets"][name] = json.loads(path.read_text())
    official_path = mongo/"public_st_history_szse.feather"
    primary = read_frame(st, nullable_booleans=True)
    official = read_frame(official_path, ["date", "code", "is_st"], nullable_booleans=True)
    combined, conflicts = merge_st_sources(primary, official)
    st_path = root/"st_history.parquet"
    combined.to_parquet(st_path, index=False)
    conflicts.to_csv(root/"st_source_disagreements.csv", index=False)
    summary["datasets"]["st_combined"] = {"output": fingerprint(st_path), "aiq": fingerprint(st),
        "official_fallback": fingerprint(official_path), "rows": len(combined),
        "codes": combined.code.nunique(), "known_event_rows": int(combined.is_st.notna().sum()),
        "conflicting_boundary_rows": len(conflicts),
        "selection": "AIQ verified states first; unknowns supplemented by official SZSE dated events; official source wins direct conflicts; all expiry NA transitions retained"}
    quote_dates = feather.read_table(repaired, columns=["date"])["date"].to_pandas()
    quote_dates = dates(quote_dates)
    status_path = root/"trade_status.parquet"
    summary["datasets"]["status"] = prepare_trade_status(database/"quote/quote.feather", database/"presaved/stoptrade.csv",
        mongo/"stock_trade_status_proxy.feather", st_path, stocks_path, calendar_path, status_path,
        quote_dates.min(), quote_dates.max())
    summary["source_selection"] = {
        "quotes": "Mongo quotefac_repaired.feather：保留已经修复的复权收益主源；不使用AIQ复权列",
        "roic": "Mongo fs_quality_fnew.feather：既有已算好的ROIC因子；AIQ没有同口径成品替代",
        "fund": "Mongo alternative_fundnum_promotion.feather：既有已算好的持仓变化因子；不以基金原始持仓代替",
        "calendar": "AIQ tradeday.csv优先；覆盖之前的日历沿用已有公开日历用于上市年龄",
        "stocks": "AIQ证券主表与明确退市日期优先；缺失证券由已有交易所主表补充；日期冲突单列",
        "history": "AIQ历史ST事件/简称交叉核验；未知回补官方SZSE历史；冲突以官方事件为准",
        "industry": "AIQ历史中信一二三级区间 → FACFUNC.get_industry → zxclassified → get_classname的35类名称",
        "float_mcap": "AIQ market_feature.circulated_market_value(万元)×10000；交易日过滤与两类公式交叉核验",
        "status": "AIQ未复权成交量/收盘价的GF停牌代理与开盘涨跌停代理优先；缺行情键沿用既有代理",
        "risk": "本次不启用Barra全中性化；旧接口保留，不作为原始/行业/市值版本门槛"}
    summary["remaining_improvements"] = [
        "AIQ旧简称历史在2022-11-25届满，2023年后上海ST仍需带事件类型和真实生效日的新来源；不以当前名称逆推历史。",
        "行业区间使用生效日及源规则，没有首次发布/修订档案；35类规则尚无完整历史版本。",
        "完整停复牌时段、开盘实际限价和特殊豁免尚不齐全，现有代理和未知继续披露。",
        "AIQ分红/股本可辅助复权检查，仍不是全公司行动与退市实际退出价值台账。",
        "AIQ直接市值与股本/代理的单源分歧须逐股复核；未知不自动补零或变成有效控制。",
        "股票池仍为沪深；北交所需要证券主表、因子、价格与交易状态同口径后单独扩展。",
        "CNE6风险模型起始日期/版本与旧Barra不同，启用全中性化前需显式字段和日期核验。"]
    assert fingerprint(repaired) == protected_price, "Repaired quote source changed during preparation"
    write_json(root/"manifest.json", summary)
    print(root/"manifest.json", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiq-root", type=Path, default=Path("/data/AIQ"))
    parser.add_argument("--mongo-root", type=Path, default=Path("/data/mongo"))
    parser.add_argument("--output", type=Path, default=Path("outputs/aiq_inputs_20261028"))
    parser.add_argument("--reuse-adapters", action="store_true", help="Reuse adapters already validated in this same preparation session")
    args = parser.parse_args()
    prepare(args.output.resolve(), args.aiq_root.resolve(), args.mongo_root.resolve(), reuse_adapters=args.reuse_adapters)
