"""Read-only adaptation of AIQ's multi-level CITIC-derived 35 classes.

The rule and name vocabulary follows Tools/Toolbox.py:zxclassified/get_classname
(without importing AIQ). Membership is evaluated over each level's effective
interval, rather than reproducing its start-date outer join and indefinite
forward fill. Missing or conflicting memberships remain unknown.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data import fingerprint, require_unique, write_json


CLASSIFICATION = "aiq_zx35"
CLASSIFICATION_VERSION = "AIQ Toolbox.py zxclassified/get_classname, 2026-10-28 snapshot"
_NAMES = (
    "银行", "证券", "保险", "房地产", "石油石化", "煤炭", "稀有金属", "基本金属", "环保事业",
    "其他公用事业", "钢铁", "基础化工", "建筑", "建材", "轻工制造", "机械", "电力设备",
    "国防军工", "汽车", "商贸零售", "消费者服务", "家电", "纺织服装", "化学制药", "太阳能及光伏",
    "新能源动力系统", "白酒", "其他食品饮料", "农林牧渔", "交通运输", "电子", "通信", "计算机", "传媒", "综合",
)
_DIRECT = {40: 1, 43: 35, 10: 5, 11: 6, 21: 11, 22: 12, 23: 13, 24: 14, 25: 15,
           28: 18, 30: 19, 31: 20, 32: 21, 33: 22, 34: 23, 35: 24, 37: 29,
           50: 30, 60: 31, 61: 32, 62: 33, 63: 34, 70: 35}
_OVERRIDES = {"600739.SH": 2, "600733.SH": 2, "600291.SH": 3, "000623.SZ": 2, "000712.SZ": 2}
_RULES = {
    1: "zx1=40", 2: "zx2=4110", 3: "zx2=4120", 4: "zx3=421010", 5: "zx1=10", 6: "zx1=11",
    7: "zx2=1230", 8: "zx1=12, excluding zx2=1230", 9: "zx2=2020",
    10: "zx1=20, excluding zx2=2020", 11: "zx1=21", 12: "zx1=22", 13: "zx1=23",
    14: "zx1=24", 15: "zx1=25", 16: "zx1=26, excluding zx3=262060",
    17: "zx1=27, excluding zx3=273030 and zx2=2740", 18: "zx1=28", 19: "zx1=30",
    20: "zx1=31", 21: "zx1=32", 22: "zx1=33", 23: "zx1=34", 24: "all zx1=35 (name retained verbatim)",
    25: "zx3=262060 or zx3=273030", 26: "zx2=2740", 27: "zx3=361010",
    28: "zx1=36, excluding zx3=361010", 29: "zx1=37", 30: "zx1=50", 31: "zx1=60",
    32: "zx1=61", 33: "zx1=62", 34: "zx1=63",
    35: "zx1=43/70; zx1=41 excluding 4110/4120; zx1=42 excluding 421010; five conditional ticker overrides",
}


def industry_catalog() -> pd.DataFrame:
    """Exact AIQ class names, including its intentionally preserved quirks."""
    result = pd.DataFrame({"class": range(1, 36), "name": _NAMES})
    result["code"] = result["class"].map(lambda value: f"AIQ_ZX_{value:02d}")
    result["custom_class_code"], result["custom_class_name"] = result["class"], result["name"]
    result["industry_code"], result["industry_name"] = result.code, result["name"]
    result["rule"] = result["class"].map(_RULES)
    result["classification"] = CLASSIFICATION
    result["classification_version"] = CLASSIFICATION_VERSION
    return result


def normalize_industry(values: pd.Series) -> pd.DataFrame:
    """Accept exact AIQ names or AIQ_ZX_01 codes, never bare CITIC numbers."""
    raw = values.astype("string")
    keys = raw.str.strip().str.casefold()
    names, codes = {}, {}
    for item in industry_catalog().itertuples(index=False):
        for alias in (item.name, item.code):
            names[alias.casefold()], codes[alias.casefold()] = item.name, item.code
    result = pd.DataFrame({"industry_raw": raw,
        "industry_code": keys.map(codes).astype("string"), "industry_name": keys.map(names).astype("string")})
    result["industry_mapping_status"] = pd.Series("mapped", index=values.index, dtype="string")
    result.loc[result.industry_name.isna(), "industry_mapping_status"] = "unmapped"
    result.loc[raw.isna() | keys.eq("").fillna(False), "industry_mapping_status"] = "missing"
    return result


normalize_aiq_industry = normalize_industry


def _classify(code: str, first: int | None, second: int | None, third: int | None) -> tuple[int | None, str, bool]:
    if first is None:
        return None, "missing_level1", False
    if ((second is not None and second // 100 != first)
            or (third is not None and third // 10000 != first)
            or (second is not None and third is not None and third // 100 != second)):
        return None, "hierarchy_mismatch", False
    if first in _DIRECT:
        value = _DIRECT[first]
    elif first in (41, 12, 20):
        if second is None:
            return None, "missing_required_level2", False
        value = ({4110: 2, 4120: 3}.get(second, 35) if first == 41 else
                 (7 if second == 1230 else 8) if first == 12 else (9 if second == 2020 else 10))
    elif first in (42, 26, 36):
        if third is None:
            return None, "missing_required_level3", False
        value = ((4 if third == 421010 else 35) if first == 42 else
                 (25 if third == 262060 else 16) if first == 26 else (27 if third == 361010 else 28))
    elif first == 27:
        if second is None:
            return None, "missing_required_level2", False
        if second == 2740:
            value = 26
        elif third is None:
            return None, "missing_required_level3", False
        else:
            value = 25 if third == 273030 else 17
    else:
        return None, "unmapped_level1", False
    override = code in _OVERRIDES and value == _OVERRIDES[code]
    return (35 if override else value), "mapped", override


def classify_aiq_industry(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify already aligned level codes; this function does no date filling."""
    result = frame.copy()
    for col in ("zx1code", "zx2code", "zx3code"):
        if col not in result:
            result[col] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    rows = [_classify(str(row.code), *(None if pd.isna(value) else int(value)
             for value in (row.zx1code, row.zx2code, row.zx3code))) for row in result.itertuples()]
    result["custom_class_code"] = pd.array([row[0] for row in rows], dtype="Int64")
    result["custom_class_name"] = result.custom_class_code.map(dict(enumerate(_NAMES, 1))).astype("string")
    result["industry_status"] = pd.array([row[1] for row in rows], dtype="string")
    result["industry_ticker_override"] = [row[2] for row in rows]
    result["industry"] = result.custom_class_name
    result["industry_name"] = result.custom_class_name
    result["industry_code"] = result.custom_class_code.map(
        lambda value: f"AIQ_ZX_{int(value):02d}" if pd.notna(value) else pd.NA).astype("string")
    result["industry_mapping_status"] = result.industry_status
    return result


def _parse_day(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    result.loc[compact] = pd.to_datetime(text[compact], format="%Y%m%d", errors="coerce")
    other = ~compact & text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    result.loc[other] = pd.to_datetime(text[other], format="%Y-%m-%d", errors="coerce")
    return result


def _prepare_levels(levels: Mapping[int, pd.DataFrame], end_inclusive: bool):
    prepared, rejected, unsafe_codes, duplicate_counts = {}, [], set(), {}
    for level in (1, 2, 3):
        source = levels[level].copy().reset_index(drop=True)
        required = {"code", "induscode", "industry", "start", "end"}
        if not required.issubset(source):
            raise ValueError(f"AIQ level {level} missing fields: {sorted(required - set(source))}")
        source["source_row"] = np.arange(len(source)) + 2
        source["code"] = source.code.astype("string").str.strip().str.upper()
        source["start_day"], source["end_day"] = _parse_day(source.start), _parse_day(source.end)
        source["sector"] = pd.to_numeric(source.induscode, errors="coerce")
        valid_code = source.code.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)", na=False)
        end_supplied = source.end.notna() & source.end.astype("string").str.strip().ne("").fillna(False)
        valid_sector = source.sector.notna() & np.isfinite(source.sector) & source.sector.mod(1).eq(0)
        valid_sector &= source.sector.between(10 ** (2 * level - 1), 10 ** (2 * level) - 1)
        reasons = pd.Series("", index=source.index, dtype="string")
        reasons.loc[~valid_code] = "invalid_code"
        reasons.loc[valid_code & ~valid_sector] = "invalid_industry_code"
        reasons.loc[valid_code & source.start_day.isna()] = "invalid_start"
        reasons.loc[valid_code & end_supplied & source.end_day.isna()] = "invalid_end"
        bad_order = source.end_day.lt(source.start_day) if end_inclusive else source.end_day.le(source.start_day)
        reasons.loc[valid_code & bad_order] = "invalid_interval_order"
        bad = reasons.ne("")
        for row_index in source.index[bad]:
            row = source.loc[row_index]
            rejected.append({"source_level": level, "source_row": int(row.source_row), "code": row.code,
                "date": row.start_day, "reason": reasons.loc[row_index], "start_raw": row.start,
                "end_raw": row.end, "induscode_raw": row.induscode})
        # An invalid interval cannot safely be bounded. Quarantine that code
        # across levels instead of letting an older valid record leak through.
        unsafe_codes.update(source.loc[bad & valid_code, "code"].tolist())
        source = source.loc[~bad].copy()
        source["sector"] = source.sector.astype("int64")
        duplicates = source.duplicated(["code", "sector", "industry", "start_day", "end_day"])
        duplicate_counts[str(level)] = int(duplicates.sum())
        source = source.loc[~duplicates].copy()
        source["stop_day"] = source.end_day + (pd.Timedelta(days=1) if end_inclusive else pd.Timedelta(0))
        prepared[level] = source
    return prepared, rejected, unsafe_codes, duplicate_counts


def build_aiq_industry_history(levels: Mapping[int, pd.DataFrame], *, end_inclusive: bool = True):
    """Return (event history, quarantine records, audit summary).

    Events at every start and termination across the three levels make an
    ordinary backward asof safe: expiry/conflict writes an explicit null row.
    End dates include that calendar day by default; an end+1 event clears it.
    """
    prepared, quarantine, unsafe_codes, duplicate_counts = _prepare_levels(levels, end_inclusive)
    grouped = {level: {str(code): frame for code, frame in data.groupby("code", sort=False)}
               for level, data in prepared.items()}
    codes = sorted(set().union(*(set(group) for group in grouped.values())))
    events = []
    for code in codes:
        frames = {level: grouped[level].get(code) for level in (1, 2, 3)}
        boundaries = sorted(set().union(*(set(frame.start_day) | set(frame.stop_day.dropna())
                              for frame in frames.values() if frame is not None)))
        for day in boundaries:
            event = {"date": day, "code": code}
            issue = "quarantined_invalid_interval" if code in unsafe_codes else None
            for level, frame in frames.items():
                active = (frame.loc[frame.start_day.le(day) & (frame.stop_day.isna() | frame.stop_day.gt(day))]
                          if frame is not None else pd.DataFrame())
                event[f"zx{level}code"] = int(active.sector.iloc[0]) if len(active) == 1 else None
                event[f"zx{level}name"] = active.industry.iloc[0] if len(active) == 1 else None
                event[f"zx{level}_active_records"] = len(active)
                if len(active) > 1:
                    issue = issue or f"overlap_level{level}"
                    quarantine.append({"source_level": level, "source_row": None, "code": code,
                        "date": day, "reason": f"overlap_level{level}", "start_raw": None,
                        "end_raw": None, "induscode_raw": ";".join(active.sector.astype(str))})
            event["_issue"] = issue
            events.append(event)
    columns = ["date", "code", *[f"zx{i}{suffix}" for i in (1, 2, 3)
               for suffix in ("code", "name", "_active_records")], "_issue"]
    history = pd.DataFrame(events, columns=columns)
    history["date"] = pd.to_datetime(history.date)
    history["code"] = history.code.astype("string")
    for level in (1, 2, 3):
        history[f"zx{level}code"] = pd.array(history[f"zx{level}code"], dtype="Int64")
        history[f"zx{level}name"] = history[f"zx{level}name"].astype("string")
    history = classify_aiq_industry(history)
    history["citic_level1_code"] = history.zx1code.astype("string")
    history["citic_level1_name"] = history.zx1name
    issue = history._issue.notna()
    history.loc[issue, "industry_status"] = history.loc[issue, "_issue"]
    invalid = history.industry_status.ne("mapped")
    for field in ("custom_class_code", "custom_class_name", "industry", "industry_name", "industry_code"):
        history.loc[invalid, field] = pd.NA
    history.loc[invalid, "industry_ticker_override"] = False
    history["industry_mapping_status"] = history.industry_status
    history["industry_source"] = CLASSIFICATION
    history = history.drop(columns="_issue").sort_values(["date", "code"]).reset_index(drop=True)
    require_unique(history, ["date", "code"])
    mismatched = history.loc[history.industry_status.eq("hierarchy_mismatch")]
    for row in mismatched.itertuples():
        quarantine.append({"source_level": None, "source_row": None, "code": row.code, "date": row.date,
            "reason": "hierarchy_mismatch", "start_raw": None, "end_raw": None,
            "induscode_raw": f"{row.zx1code}/{row.zx2code}/{row.zx3code}"})
    quarantine = pd.DataFrame(quarantine, columns=["source_level", "source_row", "code", "date", "reason",
                                                   "start_raw", "end_raw", "induscode_raw"])
    details = {"source_rows": {str(i): len(levels[i]) for i in (1, 2, 3)},
        "identical_duplicates_collapsed": duplicate_counts, "quarantined_codes": sorted(unsafe_codes),
        "status_counts": history.industry_status.value_counts().to_dict(),
        "quarantine_reason_counts": quarantine.reason.value_counts().to_dict()}
    return history, quarantine, details


def _compare_levels(levels, history, limit):
    # Independent keyed alignment, deliberately never zip/concat by CSV row.
    common = None
    for level in (1, 2, 3):
        frame = levels[level][["code", "start", "induscode"]].copy()
        frame["date"] = _parse_day(frame.start)
        frame["code"] = frame.code.astype("string").str.strip().str.upper()
        numeric = pd.to_numeric(frame.induscode, errors="coerce")
        numeric = numeric.where(np.isfinite(numeric) & numeric.mod(1).eq(0))
        numeric = numeric.where(numeric.between(10 ** (2 * level - 1), 10 ** (2 * level) - 1))
        frame[f"zx{level}code"] = numeric.astype("Int64")
        frame = frame[["date", "code", f"zx{level}code"]].dropna()
        frame = frame.loc[~frame.duplicated(["date", "code"], keep=False)]
        common = frame if common is None else common.merge(frame, on=["date", "code"], how="inner", validate="one_to_one")
    common = common.sort_values(["date", "code"])
    expected = classify_aiq_industry(common)
    joined = expected.merge(history[["date", "code", "custom_class_code", "industry_status"]],
        on=["date", "code"], how="inner", suffixes=("_source_rule", "_interval"), validate="one_to_one")
    joined["comparison_eligible"] = joined.industry_status_interval.eq("mapped") & joined.industry_status_source_rule.eq("mapped")
    joined["same_class"] = joined.custom_class_code_source_rule.eq(joined.custom_class_code_interval)
    eligible = joined.loc[joined.comparison_eligible]
    summary = {"alignment": "inner join on code and start date across all three files; independent of row order",
        "common_keys": len(common), "eligible_keys": len(eligible),
        "disagreements": int((~eligible.same_class).sum()), "sample_rows": min(limit, len(joined)),
        "limitation": "Compares keyed source-level rule evaluation to interval output; does not execute AIQ or certify vendor history."}
    return joined.head(limit), summary


def prepare_aiq_industry_history(source_dir: str | Path, output: str | Path, *,
                                 end_inclusive: bool = True, comparison_limit: int = 200) -> dict:
    """Write independent parquet, mapping, quarantine, comparison and manifest.

    ``source_dir`` can be AIQ's root or its database/sector_info directory.
    The output must be outside the input source tree (the whole AIQ directory
    when its root or database/sector_info is supplied).
    """
    source_dir, output = Path(source_dir).resolve(), Path(output).resolve()
    protected_root = source_dir
    if not (source_dir / "zx_sector1.csv").is_file():
        source_dir = source_dir / "database" / "sector_info"
    elif source_dir.name == "sector_info" and source_dir.parent.name == "database":
        protected_root = source_dir.parent.parent
    if output.suffix != ".parquet" or output.is_relative_to(protected_root):
        raise ValueError("AIQ industry export requires a separate .parquet outside its source directory")
    if comparison_limit < 0:
        raise ValueError("comparison_limit must be nonnegative")
    paths = {i: source_dir / f"zx_sector{i}.csv" for i in (1, 2, 3)}
    levels = {i: pd.read_csv(path, encoding="gb18030", dtype={"code": "string"}) for i, path in paths.items()}
    history, quarantine, details = build_aiq_industry_history(levels, end_inclusive=end_inclusive)
    comparison, compared = _compare_levels(levels, history, comparison_limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    try:
        history.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    mapping_path = output.with_name(output.stem + "_mapping.csv")
    quarantine_path = output.with_name(output.stem + "_quarantine.csv")
    comparison_path = output.with_name(output.stem + "_comparison.csv")
    industry_catalog().to_csv(mapping_path, index=False, encoding="utf-8-sig")
    quarantine.to_csv(quarantine_path, index=False, encoding="utf-8-sig")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    rule_source = source_dir.parent.parent / "Tools" / "Toolbox.py"
    summary = {"classification": CLASSIFICATION, "classification_version": CLASSIFICATION_VERSION,
        "rule_reference": "Tools/Toolbox.py:95 zxclassified and :200 get_classname; FAC_METHOD.py:368 get_industry",
        "rule_source_fingerprint": fingerprint(rule_source) if rule_source.is_file() else None,
        "adapter_fingerprint": fingerprint(Path(__file__).resolve()),
        "source_fingerprints": {str(i): fingerprint(path) for i, path in paths.items()},
        "output_fingerprint": fingerprint(output), "rows": len(history), "codes": int(history.code.nunique()),
        "start": history.date.min(), "end": history.date.max(), **details,
        "date_policy": "start effective on stated day; end inclusive, clear at end+1" if end_inclusive
            else "start inclusive, end exclusive; clear at end",
        "end_inclusive": end_inclusive, "no_date_backshift": True,
        "overlap_policy": "conflicting simultaneous intervals remain unknown until unambiguous; explicit null events block forward carry",
        "invalid_interval_policy": "quarantine the entire code when an invalid interval cannot be bounded safely",
        "ticker_overrides": _OVERRIDES, "ticker_override_rows": int(history.industry_ticker_override.sum()),
        "comparison": compared, "mapping_path": str(mapping_path), "quarantine_path": str(quarantine_path),
        "comparison_path": str(comparison_path),
        "limitations": ["Static AIQ rules have no historical version dates and are not PIT-certified classification standards.",
            "Class 24 is named 化学制药 in get_classname but maps all level-one 医药, exactly as current source code.",
            "Class 35 mixes nonbank/real-estate remainders, 综合金融, 综合 and five conditional stock overrides.",
            "End inclusion is an explicit adapter convention, supported by adjacent source intervals; vendor publication timing is unverified.",
            "Missing required sublevels are unknown instead of being assigned to a broad residual class.",
            "This adapter does not download, import AIQ modules, alter source history or use the obsolete industryclass.xlsx."]}
    write_json(output.with_suffix(".manifest.json"), summary)
    return summary
