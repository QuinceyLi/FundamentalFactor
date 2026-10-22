"""Fetch/validate official Shenzhen name events and stage historical ST states.

Source directory writes are deliberately left to the publishing step.  Neither
an unavailable vendor API nor a current Shanghai list is treated as historical
coverage.  ``--download`` refreshes the public exchange response; otherwise this
command deterministically rebuilds the staged files from its recorded response.
"""
from __future__ import annotations

import argparse
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
from fundamental_factor.data import dates, write_json
from fundamental_factor.st_history import exchange_st_events, normalize_szse_names, st_from_names


SZSE_URL = "https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=SSGSGMXX&TABKEY=tab2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(root):
    import requests
    response = requests.get(SZSE_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / "szse_name_changes.xlsx"
    path.write_bytes(response.content)
    write_json(raw / "szse_fetch_manifest.json", {"url": SZSE_URL, "file": path.name,
        "http_status": response.status_code, "sha256": sha(path),
        "collected_utc": dt.datetime.now(dt.timezone.utc).isoformat()})


def stage(root: Path, exposure_path: Path | None = None):
    raw, output = root / "raw", root / "staging"
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((raw / "szse_fetch_manifest.json").read_text())
    path = raw / source["file"]
    if source["http_status"] != 200 or sha(path) != source["sha256"]:
        raise ValueError("Downloaded SZSE response failed provenance verification")
    observed = pd.Timestamp(source["collected_utc"]).tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        source_frame = pd.read_excel(path, dtype=str)
    names = normalize_szse_names(source_frame, observed)
    states = exchange_st_events(names, observed)
    datasets = []
    for filename, frame in [("public_name_change_szse_validated.feather", names),
                            ("public_st_history_szse.feather", states)]:
        frame = frame.copy()
        for column in ("date", "observed_date", "valid_until"):
            if column in frame:
                frame[column] = frame[column].dt.strftime("%Y%m%d").astype("int64")
        target = output / filename
        frame.reset_index(drop=True).to_feather(target)
        restored = pd.read_feather(target)
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), restored)
        datasets.append({"file": filename, "rows": len(frame), "codes": frame.code.nunique(),
            "date_min": int(frame.date.min()), "date_max": int(frame.date.max()),
            "is_st_true": int(frame.is_st.sum()) if "is_st" in frame else None,
            "unknown_rows": int(frame.is_st.isna().sum()) if "is_st" in frame else None,
            "sha256": sha(target), "columns": frame.columns.tolist()})
    comparison = {"status": "not_requested"}
    if exposure_path is not None and exposure_path.exists():
        table = feather.read_table(exposure_path, columns=["date", "code", "secShortName"])
        sample_dates = [20240102, 20240401, 20240701, 20241008, 20250102, 20250401,
                        20250701, 20251009, 20260105, 20260401, 20260701, 20260803]
        query_dates = (pa.array(pd.to_datetime(pd.Series(sample_dates).astype(str), format="%Y%m%d"))
                       if pa.types.is_timestamp(table.schema.field("date").type) else pa.array(sample_dates))
        table = table.filter(pc.is_in(table["date"], value_set=query_dates))
        sample = table.to_pandas()
        sample = sample[sample.code.str.endswith(".SZ")].copy()
        sample["date"] = dates(sample.date)
        sample["local_name_is_st"] = st_from_names(sample.secShortName)
        joined = pd.merge_asof(sample.sort_values("date"),
            states[["date", "code", "is_st", "name_after"]].sort_values("date"),
            by="code", on="date", direction="backward")
        known = joined[joined.is_st.notna() & joined.local_name_is_st.notna()].copy()
        if known.empty:
            raise ValueError("Requested local-name cross-check produced no comparable observations")
        mismatch = known[known.local_name_is_st.ne(known.is_st)]
        mismatch.to_csv(output / "local_name_proxy_disagreements.csv", index=False)
        comparison = {"status": "complete", "local_file": str(exposure_path),
            "sampling_dates": sample_dates, "compared_known_records": len(known),
            "different_st_records": len(mismatch), "different_codes": mismatch.code.nunique(),
            "interpretation": "本地 secShortName 是待核验代理；与官方生效事件冲突时采用官方事件，不把本地名称回填为真实历史。"}
    old_name_pattern = names.name_after.str.replace(r"\s+", "", regex=True).str.upper().str.match(r"^(?:S\*?ST|\*?ST)")
    corrected = names[names.st_in_name_after.ne(old_name_pattern)]
    corrected.to_csv(output / "historical_prefix_corrections.csv", index=False)
    failures = []
    connection_log = raw / "baostock_connectivity.json"
    if connection_log.exists():
        failures = json.loads(connection_log.read_text())
    manifest = {
        "source": source, "observed_date": observed.strftime("%Y-%m-%d"),
        "source_rows_all_markets": len(source_frame), "name_events_a_share": len(names),
        "name_chains_checked": names.code.nunique(), "name_chain_discontinuities": 0,
        "historical_g_prefix_corrections": len(corrected), "datasets": datasets,
        "local_name_crosscheck": comparison, "baostock_connection_attempts": failures,
        "truth_boundary": [
            "ST状态由深交所官方带生效日期的变更后简称确定，涵盖ST、*ST、SST、S*ST、GST、G*ST；不是停牌代理。",
            "首次事件之前为未知；没有事件的股票为未知；禁止用未来变更前名称逆推或用当前名单回填。",
            "事件后延续至下一事件，基于本次官方全量名称变更表；通过逐股前后简称链核验，仍不宣称官方源绝无遗漏。",
            "只在生效日及以后使用事件，不假定获得了更早的公告日；股票有效生命周期由股票主表过滤。",
            "采集日下一自然日写入未知哨兵，防止读取此冻结版本时向未来无限延续。",
            "上海历史尚未获得可靠公开全量源，BaoStock两次指定公开服务器连接均失败；Sina名字列表缺日期，上交所风险板是当前列表，均未转换为历史ST。",
        ],
        "publication_directory": "/data/mongo", "formal_full_sh_sz_history": False,
    }
    write_json(output / "public_st_history_manifest.json", manifest)
    print(json.dumps({"datasets": datasets, "crosscheck": comparison,
                       "historical_prefix_corrections": len(corrected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/historical_st")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--exposure", type=Path, default=Path("/home/liqinghua/data/combined_exp.feather"))
    args = parser.parse_args()
    if args.download:
        fetch(args.root)
    stage(args.root, args.exposure)
