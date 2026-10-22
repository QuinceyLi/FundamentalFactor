"""Validate downloaded public records and stage new Feather files for publication.

No network or writes to source directories. Calendar decoding is performed with
the pinned AKShare decoder beforehand; this script only publishes its locally
cross-checked interval. Source responses and full SHA256 remain in the manifest.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import warnings

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fundamental_factor.data import dates, require_unique, write_json


def compact(values):
    return dates(pd.Series(values).reset_index(drop=True)).dt.strftime("%Y%m%d").astype("int64")


def prepare(root: Path):
    raw, target = root / "raw", root / "validated"
    target.mkdir(exist_ok=True, parents=True)
    sources = []
    for name in ("fetch_manifest.json", "szse_fetch_manifest.json", "sse_fetch_manifest.json"):
        rows = json.loads((raw / name).read_text())
        for row in rows:
            if row.get("http_status") != 200 or hashlib.sha256((raw / row["file"]).read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Source fetch/hash not verified: {row['file']}")
            sources.append(row)
    collected = int(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d"))
    summaries = []

    def emit(name, frame, keys, scope):
        frame = frame.sort_values(keys).reset_index(drop=True)
        require_unique(frame, keys)
        if "code" in frame and not frame.code.str.fullmatch(r"\d{6}\.(SH|SZ)").all():
            raise ValueError("Noncanonical Shanghai/Shenzhen stock code")
        if frame.empty:
            raise ValueError("Empty dataset cannot be published")
        path = target / name
        frame.to_feather(path)
        check = feather.read_table(path)
        if check.num_rows != len(frame) or not pa.types.is_int64(check.schema.field("date").type):
            raise ValueError("Feather round-trip/schema failure")
        summaries.append({"file": name, "rows": len(frame), "date_min": int(frame.date.min()),
            "date_max": int(frame.date.max()), "keys": keys, "columns": frame.columns.tolist(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "scope": scope})
        return frame

    decoded = pd.read_parquet(root / "calendar_decoded.parquet").date
    require_unique(decoded.to_frame("date"), ["date"])
    quote_days = pd.read_parquet(root.parent / "gfoptim_inputs/calendar_candidates.parquet").date
    start, end = max(decoded.min(), quote_days.min()), min(decoded.max(), quote_days.max())
    public = decoded[decoded.between(start, end)].sort_values().reset_index(drop=True)
    local = quote_days[quote_days.between(start, end)].sort_values().reset_index(drop=True)
    if public.empty or not public.equals(local):
        raise ValueError("Sina calendar disagrees with independent local quote dates")
    emit("public_trade_calendar_sina.feather", pd.DataFrame({"date": compact(public)}), ["date"],
         "仅保存与本地原始行情日期逐日一致的区间；不发布未核验1990年代记录或未来计划日历")

    def excel(name):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Workbook contains no default style")
            return pd.read_excel(raw / name, dtype=str)

    # A-share codes are selected from the exchange's dedicated A-share columns.
    active = excel("szse_listed.xlsx")
    active = active[active["A股代码"].str.fullmatch(r"[03]\d{5}", na=False)].reset_index(drop=True)
    sz_active = pd.DataFrame({"date": compact(active["A股上市日期"]), "code": active["A股代码"] + ".SZ",
        "name_observed": active["A股简称"], "listing_date": compact(active["A股上市日期"]),
        "source": "SZSE", "observed_date": collected, "observed_status": "listed"})
    dead = excel("szse_delisted.xlsx")
    dead = dead[dead["证券代码"].str.fullmatch(r"[03]\d{5}", na=False)].reset_index(drop=True)
    sz_dead = pd.DataFrame({"date": compact(dead["终止上市日期"]), "code": dead["证券代码"] + ".SZ",
        "name_observed": dead["证券简称"], "listing_date": compact(dead["上市日期"]),
        "source": "SZSE", "observed_date": collected})
    sh_lists, sh_deads = [], []
    for name in ("sse_main.json", "sse_star.json", "sse_delisted.json"):
        response = json.loads((raw / name).read_text())
        frame = pd.DataFrame(response["result"])
        if len(frame) != int(response["pageHelp"]["total"]):
            raise ValueError(f"Pagination incomplete: {name}")
        if not frame.A_STOCK_CODE.str.fullmatch(r"6\d{5}").all():
            raise ValueError("SSE endpoint did not return exclusively A shares")
        block = pd.DataFrame({"date": compact(frame.LIST_DATE), "code": frame.A_STOCK_CODE + ".SH",
            "name_observed": frame.SEC_NAME_CN.where(frame.SEC_NAME_CN.ne("-"), frame.COMPANY_ABBR),
            "listing_date": compact(frame.LIST_DATE), "source": "SSE", "observed_date": collected})
        if name == "sse_delisted.json":
            block["date"] = compact(frame.DELIST_DATE)
            sh_deads.append(block)
        else:
            block["observed_status"] = "listed"
            sh_lists.append(block)
    delisted = pd.concat([sz_dead, *sh_deads], ignore_index=True)
    if (delisted.date < delisted.listing_date).any():
        raise ValueError("Delisting precedes listing")
    emit("public_stock_delisting_sh_sz.feather", delisted, ["date", "code"],
         "沪深交易所本次返回的终止上市记录；仅日期事实，不代表最后交易日、清算或每股现金")
    historical = delisted.copy()
    historical["date"] = historical.listing_date
    historical["observed_status"] = "delisted"
    listings = pd.concat([sz_active, *sh_lists, historical], ignore_index=True)
    emit("public_stock_listing_sh_sz.feather", listings, ["date", "code"],
         "现上市与终止上市列表包含的原始上市日期；name_observed为采集时显示名称，不用于历史ST回填")
    require_unique(listings, ["code"])
    require_unique(delisted, ["code"])
    master = listings[["date", "code", "listing_date", "source", "observed_date"]].merge(
        delisted[["code", "date"]].rename(columns={"date": "delisting_date"}), on="code", how="left", validate="one_to_one")
    master["delisting_date"] = master.delisting_date.astype("Int64")
    master["is_a_share"] = True
    emit("public_stock_master_sh_sz.feather", master, ["code"],
         "官方在市与已退市A股记录合并；上市/终止日期用于股票池，未知ST不推断；暂不表示重新上市等完整证券生命周期")

    names = excel("szse_name_changes.xlsx")
    names = names[names["证券代码"].str.fullmatch(r"[03]\d{5}", na=False)].reset_index(drop=True)
    history = pd.DataFrame({"date": compact(names["变更日期"]), "code": names["证券代码"] + ".SZ",
        "name_before": names["变更前简称"], "name_after": names["变更后简称"],
        "source": "SZSE", "observed_date": collected})
    for endpoint in ("before", "after"):
        history[f"st_in_name_{endpoint}"] = (history[f"name_{endpoint}"].str.replace(r"\s", "", regex=True)
            .str.upper().str.match(r"^(?:S\*?ST|\*?ST)", na=False))
    emit("public_name_change_szse.feather", history, ["date", "code"],
         "深交所A股历史简称变更；ST标记只解释对应事件的名称，未将无记录股票/区间补为非ST；不含上海")
    codes = set(listings.code)
    quote_codes = set(pc.unique(feather.read_table("/data/mongo/quotefac.feather", columns=["code"])["code"]).to_pylist())
    missing = sorted(quote_codes - codes)
    # An incomplete exchange response must not silently become the project's complete master table.
    manifest = {"collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "format": "Feather V2; date int64 YYYYMMDD; code six-digit + .SH/.SZ",
        "publish_directory": "/data/mongo", "source_responses": sources, "datasets": summaries,
        "calendar_crosscheck": {"start": start, "end": end, "matched_dates": len(public), "sina_only": 0, "local_only": 0},
        "metadata_coverage": {"listed_or_delisted_codes": len(codes), "quote_codes": len(quote_codes),
                              "missing_quote_codes": missing},
        "remaining": ["完整沪深历史ST状态（本次仅获得深圳名称事件）", "行业class分类定义和class35含义/时点确认",
            "历史逐股开盘涨跌停及无价格限制例外、停牌代理核验", "完整公司行动与已确认复权异常修复",
            "实际持仓涉及的退市后续估值/退出事件；退市不自动等于现金清算"],
        "formal_backtest_ready": False,
        "truth_scope": "只保存真实来源响应及确定性转换；来源完整性、未知区间与结算金额不作推断，未覆盖字段不补零"}
    write_json(target / "public_reference_manifest.json", manifest)
    print(json.dumps({"datasets": summaries, "metadata_coverage": manifest["metadata_coverage"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1] / "outputs/public_data_review")
