"""CITIC level-one labels and auditable conversion of vendor one-hot history.

The native on_all_temp classes 1..35 are deliberately not aliases for CITIC.
Classification names describe a fixed 2.0 vocabulary; stock memberships must
come from dated source observations, never from today's constituent list.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as parquet

from .data import dates, fingerprint, require_unique, write_json


CLASSIFICATION = "citic_l1"
CLASSIFICATION_VERSION = "CITIC 2.0 (2020), historical vendor membership"
# Original CITIC Securities research, table 5 (2020-01-02), mirrored full text.
MAPPING_SOURCE = ("https://www.scribd.com/document/659446350/"
                  "%E4%B8%AD%E4%BF%A1%E8%AF%81%E5%88%B8%E8%A1%8C%E4%B8%9A%E5%88%86%E7%B1%BB%E6%A0%87%E5%87%86-2-0-"
                  "%E7%89%88%E5%8F%8A%E4%BF%AE%E8%AE%A2%E8%AF%B4%E6%98%8E-2020-1-2")
MAPPING_REFERENCE = "中信证券研究部《行业分类标准2.0版及修订说明》2020-01-02，表5"
# (official two-digit sector code, index code, Chinese label, vendor field)
_SECTORS = (
    ("10", "CI005001", "石油石化", "Petroleum"),
    ("11", "CI005002", "煤炭", "Coal"),
    ("12", "CI005003", "有色金属", "NonferrousMetals"),
    ("20", "CI005004", "电力及公用事业", "PowerUtilities"),
    ("21", "CI005005", "钢铁", "Steel"),
    ("22", "CI005006", "基础化工", "BasicChemicals"),
    ("23", "CI005007", "建筑", "Construction"),
    ("24", "CI005008", "建材", "BuildMater"),
    ("25", "CI005009", "轻工制造", "LightIndustry"),
    ("26", "CI005010", "机械", "Machinery"),
    ("27", "CI005011", "电力设备及新能源", "PowerEquipNewEnergy"),
    ("28", "CI005012", "国防军工", "Defense"),
    ("30", "CI005013", "汽车", "Automobiles"),
    ("31", "CI005014", "商贸零售", "RetailTrade"),
    ("32", "CI005015", "消费者服务", "ConsumerServices"),
    ("33", "CI005016", "家电", "HomeAppliances"),
    ("34", "CI005017", "纺织服装", "TextileGarment"),
    ("35", "CI005018", "医药", "HealthCare"),
    ("36", "CI005019", "食品饮料", "FoodBeverages"),
    ("37", "CI005020", "农林牧渔", "Agriculture"),
    ("40", "CI005021", "银行", "Banks"),
    ("41", "CI005022", "非银行金融", "NonbankFinan"),
    ("42", "CI005023", "房地产", "RealEstate"),
    ("43", "CI005030", "综合金融", "DiverseFinan"),
    ("50", "CI005024", "交通运输", "Transportation"),
    ("60", "CI005025", "电子", "Electronics"),
    ("61", "CI005026", "通信", "Telecoms"),
    ("62", "CI005027", "计算机", "Computers"),
    ("63", "CI005028", "传媒", "Media"),
    ("70", "CI005029", "综合", "Conglomerates"),
)
INDUSTRY_FIELDS = tuple(row[3] for row in _SECTORS)
LEGACY_FIELDS = {"ConsumerServices": "CateringTourism", "Electronics": "ElectronicCompon",
                 "PowerEquipNewEnergy": "PowerEquip"}
_NAME_ALIASES = {"餐饮旅游": "消费者服务", "电子元器件": "电子", "电力设备": "电力设备及新能源",
                 "非银金融": "非银行金融"}


def industry_catalog() -> pd.DataFrame:
    """Return the field/name/code dictionary; these rows are not memberships."""
    result = pd.DataFrame(_SECTORS, columns=["citic_level1_code", "industry_code", "industry_name", "industry_field"])
    result["legacy_field"] = result.industry_field.map(LEGACY_FIELDS).fillna("")
    result["classification"] = CLASSIFICATION
    result["classification_version"] = CLASSIFICATION_VERSION
    result["mapping_reference"] = MAPPING_REFERENCE
    result["mapping_source"] = MAPPING_SOURCE
    return result


def normalize_industry(values: pd.Series) -> pd.DataFrame:
    """Normalize recognized labels/codes, preserving unknown native categories.

    Bare numbers are not accepted: source-specific class_35 is not CITIC's
    two-digit code 35. Explicit Chinese/English labels and CI index codes are
    unambiguous within this explicitly selected classification.
    """
    raw = values.astype("string")
    keys = raw.str.strip().str.casefold()
    names, codes = {}, {}
    for _, code, name, field in _SECTORS:
        aliases = [code, code + ".WI", name, field, name + "(中信)", name + "（中信）"]
        if field in LEGACY_FIELDS:
            aliases.append(LEGACY_FIELDS[field])
        aliases.extend(alias for alias, target in _NAME_ALIASES.items() if target == name)
        for alias in aliases:
            names[alias.casefold()], codes[alias.casefold()] = name, code
    result = pd.DataFrame({"industry_raw": raw, "industry_code": keys.map(codes).astype("string"),
                           "industry_name": keys.map(names).astype("string")})
    result["industry_mapping_status"] = pd.Series("mapped", index=values.index, dtype="string")
    missing = raw.isna() | keys.eq("").fillna(False)
    result.loc[result.industry_name.isna(), "industry_mapping_status"] = "unmapped"
    result.loc[missing, "industry_mapping_status"] = "missing"
    return result


def industry_from_exposures(frame: pd.DataFrame) -> pd.DataFrame:
    """Decode exactly one finite binary industry; conflicting aliases stay NA."""
    resolved = pd.DataFrame(index=frame.index)
    conflict = pd.Series(False, index=frame.index)
    for field in INDUSTRY_FIELDS:
        alternatives = [name for name in (field, LEGACY_FIELDS.get(field)) if name in frame]
        if not alternatives:
            raise ValueError(f"Missing CITIC one-hot field: {field}")
        value = pd.to_numeric(frame[alternatives[0]], errors="raise")
        for alternative in alternatives[1:]:
            other = pd.to_numeric(frame[alternative], errors="raise")
            conflict |= value.notna() & other.notna() & value.ne(other)
            value = value.combine_first(other)
        resolved[field] = value
    # Some vendor files retain obsolete columns, permanently zero, beside the
    # complete current 30-column schema. A valid current one-hot vector is
    # authoritative; old fields are fallbacks only for missing current values.
    canonical_valid = pd.Series(False, index=frame.index)
    if set(INDUSTRY_FIELDS).issubset(frame):
        current = frame[list(INDUSTRY_FIELDS)]
        canonical_valid = current.isin([0, 1]).all(axis=1) & current.sum(axis=1).eq(1)
    good = (resolved.isin([0, 1]).all(axis=1) & resolved.sum(axis=1).eq(1) & ~conflict) | canonical_valid
    selected = resolved.fillna(0).idxmax(axis=1).astype("string").where(good)
    result = frame[["date", "code"]].copy()
    result["date"] = dates(result.date)
    result["code"] = result.code.astype("string").str.strip().str.upper()
    result["code"] = result.code.str.replace(".XSHE", ".SZ", regex=False).str.replace(".XSHG", ".SH", regex=False)
    if not result.code.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)", na=False).all():
        raise ValueError("Noncanonical A-share code in industry source")
    normalized = normalize_industry(selected)
    for col in normalized:
        result[col] = normalized[col]
    result.loc[~good, "industry_mapping_status"] = "invalid_onehot"
    result.loc[conflict & ~good, "industry_mapping_status"] = "conflicting_aliases"
    result["industry"] = result.industry_name
    result["industry_onehot_valid"] = good
    return result


def prepare_industry_history(source: str | Path, output: str | Path,
                             comparison_source: str | Path | None = None,
                             *, update_time_timezone: str | None = None) -> dict:
    """Export on the recorded update day, preserving observation and update time.

    Vendor updateTime is not certified first publication. For naive timestamps
    with unknown timezone, add a calendar-day buffer to max(observation day,
    recorded update day). An explicitly verified UTC/Shanghai timezone permits
    using the converted Shanghai update day without this extra buffer.
    Missing/invalid timestamps are quarantined; same-day updates retain the
    newest observation. No source value is modified.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.suffix != ".parquet":
        raise ValueError("Industry export requires a separate .parquet output")
    if update_time_timezone not in (None, "UTC", "Asia/Shanghai"):
        raise ValueError("update_time_timezone must be verified UTC/Asia/Shanghai, or None")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    writer = None
    quarantine = []
    source_rows = 0
    try:
        with pa.memory_map(str(source), "r") as stream:
            reader = ipc.open_file(stream)
            fields = [name for name in reader.schema.names if name in
                      {"date", "code", "updateTime", *INDUSTRY_FIELDS, *LEGACY_FIELDS.values()}]
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i).select(fields).to_pandas()
                normalized = industry_from_exposures(batch)
                source_rows += len(normalized)
                normalized["industry_observation_date"] = normalized.date
                normalized["industry_update_time"] = (pd.to_datetime(batch.updateTime, errors="coerce")
                                                       if "updateTime" in batch else pd.NaT)
                missing_update = normalized.industry_update_time.isna()
                if missing_update.any():
                    rejected = normalized.loc[missing_update, ["code", "industry_observation_date", "industry_raw"]].copy()
                    rejected["reason"] = "missing_or_invalid_update_time"
                    quarantine.append(rejected)
                normalized = normalized.loc[~missing_update].copy()
                local_update = normalized.industry_update_time
                if getattr(local_update.dt, "tz", None) is not None:
                    local_update = local_update.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
                    extra_day = 0
                elif update_time_timezone is not None:
                    local_update = local_update.dt.tz_localize(update_time_timezone).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
                    extra_day = 0
                else:
                    extra_day = 1
                normalized["date"] = pd.concat([normalized.industry_observation_date,
                    local_update.dt.normalize()], axis=1).max(axis=1) + pd.Timedelta(days=extra_day)
                table = pa.Table.from_pandas(normalized, preserve_index=False)
                if writer is None:
                    writer = parquet.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        if writer is None:
            raise ValueError("Empty industry source")
        writer.close()
        writer = None
        keys = parquet.read_table(temporary, columns=["date", "code", "industry_observation_date", "industry_update_time"]).to_pandas()
        # A duplicate original snapshot is a source error, even if sorting
        # could otherwise select one of its conflicting labels arbitrarily.
        require_unique(keys, ["industry_observation_date", "code"])
        selected = keys.sort_values(["date", "code", "industry_observation_date", "industry_update_time"]).drop_duplicates(["date", "code"], keep="last")
        same_day_consolidated = len(keys) - len(selected)
        newest_known = selected.groupby("code").industry_observation_date.cummax()
        stale = selected.industry_observation_date.lt(newest_known)
        stale_count = int(stale.sum())
        if stale_count:
            rejected = selected.loc[stale, ["code", "industry_observation_date", "industry_update_time"]].copy()
            rejected["reason"] = "stale_late_observation"
            quarantine.append(rejected)
            selected = selected.loc[~stale]
        export = parquet.read_table(temporary).take(pa.array(selected.index.to_numpy()))
        parquet.write_table(export, output, compression="zstd")
        del export
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    normalized = pd.read_parquet(output, columns=["date", "code", "industry", "industry_mapping_status"])
    summary = {"classification": CLASSIFICATION, "classification_version": CLASSIFICATION_VERSION,
               "mapping_reference": MAPPING_REFERENCE, "mapping_source": MAPPING_SOURCE,
               "source_fingerprint": fingerprint(source), "output_fingerprint": fingerprint(output),
               "source_rows": source_rows, "rows": len(normalized), "codes": normalized.code.nunique(),
               "start": normalized.date.min(), "end": normalized.date.max(),
               "observation_start": keys.industry_observation_date.min(),
               "observation_end": keys.industry_observation_date.max(),
               "same_day_updates_consolidated": same_day_consolidated,
               "stale_late_observations_ignored": stale_count,
               "quarantined_missing_update_time": sum(int(frame.reason.eq("missing_or_invalid_update_time").sum()) for frame in quarantine),
               "date_policy": "max(observation_date, update day) + 1 calendar day if timezone unknown; newest observation per code/day",
               "update_time_timezone": update_time_timezone or "unverified; naive recorded clock with one-day conservative buffer",
               "mapping_status": normalized.industry_mapping_status.value_counts().to_dict(),
               "industry_counts": normalized.industry.value_counts().to_dict(),
               "limitations": ["Vendor historical classification; not an exchange-certified full history.",
                               "No class_1..35 inference and no current-membership backfill.",
                               "date uses the recorded update day as a conservative availability proxy; it is not certified first publication.",
                               "Naive updateTime without a verified timezone is delayed one additional calendar day.",
                               "Observation days before the first known update are not backfilled."]}
    if comparison_source is not None:
        summary["comparison"] = compare_industry_sample(output, Path(comparison_source))
    industry_catalog().to_csv(output.parent / "industry_mapping.csv", index=False)
    quarantined = (pd.concat(quarantine, ignore_index=True) if quarantine else
                   pd.DataFrame(columns=["code", "industry_observation_date", "industry_raw", "reason"]))
    quarantined.to_csv(output.parent / "industry_quarantine.csv", index=False)
    write_json(output.with_suffix(".manifest.json"), summary)
    return summary


def compare_industry_sample(normalized_path: Path, comparison_source: Path) -> dict:
    """Cross-check twelve historical dates, without replacing any observation."""
    available = feather.read_table(comparison_source, columns=["date"]).to_pandas().date
    unique = np.sort(pd.to_datetime(available).unique())
    selected = unique[np.unique(np.linspace(0, len(unique) - 1, min(12, len(unique)), dtype=int))]
    with pa.memory_map(str(comparison_source), "r") as stream:
        reader = ipc.open_file(stream)
        fields = [name for name in reader.schema.names if name in {"date", "code", *INDUSTRY_FIELDS, *LEGACY_FIELDS.values()}]
        blocks = []
        for i in range(reader.num_record_batches):
            chunk = reader.get_batch(i).select(fields).to_pandas()
            chunk = chunk[chunk.date.isin(selected)]
            if not chunk.empty:
                blocks.append(industry_from_exposures(chunk)[["date", "code", "industry_code"]])
    reference = pd.concat(blocks, ignore_index=True)
    require_unique(reference, ["date", "code"])
    primary = pd.read_parquet(normalized_path, columns=["industry_observation_date", "code", "industry_code"],
                              filters=[("industry_observation_date", "in", [pd.Timestamp(day) for day in selected])])
    primary = primary.rename(columns={"industry_observation_date": "date"})
    compared = primary.merge(reference, on=["date", "code"], how="outer", suffixes=("_primary", "_comparison"), indicator=True)
    both = compared._merge.eq("both")
    mismatch = both & compared.industry_code_primary.fillna("").ne(compared.industry_code_comparison.fillna(""))
    return {"source_fingerprint": fingerprint(comparison_source), "sample_dates": [pd.Timestamp(x) for x in selected],
            "matched_rows": int(both.sum()), "unmatched_primary": int(compared._merge.eq("left_only").sum()),
            "unmatched_comparison": int(compared._merge.eq("right_only").sum()),
            "disagreements": int(mismatch.sum()), "disagreement_examples": compared.loc[mismatch].head(20).to_dict("records"),
            "method": "same date/code exact classification; disclose only, no replacement"}
