"""Point-in-time quote-implied floating market capitalization, in yuan.

This is an explicitly selected proxy, not a replacement for verified share
capital. Raw close/volume and turnover must describe the same stock-day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import asof_snapshot, columns, read_frame


PROXY_COLUMNS = ["market_cap_proxy", "market_cap_date", "market_cap_quote_date",
                 "market_cap_turnover_date", "market_cap_close", "market_cap_volume",
                 "market_cap_turnover", "market_cap_invalid_reason"]

HISTORY_COLUMNS = ["market_cap_date", "market_cap_source", "market_cap_invalid_reason",
                   "market_cap_status", "market_cap_crosscheck_status",
                   "float_mcap_original", "float_mcap_original_missing", "float_mcap_original_invalid",
                   "float_mcap_filled", "float_mcap_value_source_date", "float_mcap_fill_anchor_date",
                   "market_cap_grid_date", "market_cap_effective_status"]


def historical_market_cap_snapshot(config: Config, base: pd.DataFrame) -> pd.DataFrame:
    """Read a separate yuan-valued market-cap feed, preserving explicit null updates."""
    spec = config.market_cap
    path = config.source(spec["source"])
    result = base.drop(columns="float_mcap", errors="ignore").copy()
    if path is None or not path.is_file():
        if not config.optional_control_sources:
            raise ValueError("missing historical floating market-cap source")
        result["float_mcap"] = np.nan
        result["market_cap_date"] = pd.NaT
        result["market_cap_invalid_reason"] = "missing_source"
        return result
    available = columns(path)
    field = spec["column"]
    provenance = [name for name in HISTORY_COLUMNS if name != "market_cap_date" and name in available]
    selected = ["date", "code"] + ([field] if field in available else []) + provenance
    if config.aiq_fill_enabled and "raw_mcap_wan" in available:
        selected.append("raw_mcap_wan")
    source = read_frame(path, list(dict.fromkeys(selected)))
    if field not in source:
        if not config.optional_control_sources:
            raise ValueError(f"historical market-cap source missing {field}")
        source[field] = np.nan
    source = source.rename(columns={field: "float_mcap"})
    source["float_mcap"] = pd.to_numeric(source.float_mcap, errors="coerce")
    if config.aiq_fill_enabled and "raw_mcap_wan" in source:
        # The validated adapter retains raw values. AIQ fills only NA, so a
        # source zero/negative/infinity must not turn into an older positive cap.
        recover = source.market_cap_status.eq("nonfinite_or_nonpositive") if "market_cap_status" in source else pd.Series(True, index=source.index)
        source.loc[recover, "float_mcap"] = pd.to_numeric(source.loc[recover, "raw_mcap_wan"], errors="coerce") * 10000.
    invalid = ~np.isfinite(source.float_mcap) | source.float_mcap.le(0)
    if not config.aiq_fill_enabled:
        source.loc[invalid, "float_mcap"] = np.nan
    if "market_cap_invalid_reason" not in source:
        source["market_cap_invalid_reason"] = ""
    reason_missing = source.market_cap_invalid_reason.fillna("").eq("")
    source.loc[invalid & reason_missing, "market_cap_invalid_reason"] = "invalid_historical_value"
    source = source[source.date.le(base.cutoff.max()) & source.code.isin(base.code.unique())]
    if not config.aiq_fill_enabled:
        return asof_snapshot(result, source, "market_cap")
    from .aiq_fill import filled_snapshot
    result = filled_snapshot(result, source, ["float_mcap"], prefix="market_cap",
                             calendar=read_frame(config.source("calendar"), ["date"]).date)
    original = pd.to_numeric(result.float_mcap_original, errors="coerce")
    result["float_mcap_original_invalid"] = ~np.isfinite(original) | original.le(0)
    invalid = ~np.isfinite(result.float_mcap) | result.float_mcap.le(0)
    result.loc[invalid, "float_mcap"] = np.nan
    result["market_cap_effective_status"] = np.where(invalid, "unavailable", np.where(result.float_mcap_filled, "aiq_filled", "observed"))
    return result


def quote_market_cap(close: pd.Series, volume: pd.Series, turnover: pd.Series) -> pd.DataFrame:
    """Compute close * volume(lots) * 100 / (turnover(percent) / 100).

    Reject nonpositive/nonfinite inputs and overflow; do not impute zero or
    carry an older valid observation over an explicit invalid update.
    """
    values = pd.DataFrame({"close": close, "volume": volume, "turnover": turnover}).astype(float)
    valid = pd.Series(True, index=values.index)
    reason = pd.Series("", index=values.index, dtype="string")
    for name in values:
        bad = ~np.isfinite(values[name]) | values[name].le(0)
        valid &= ~bad
        reason.loc[bad] += f"invalid_{name};"
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        proxy = values.close * values.volume * 100. / (values.turnover / 100.)
    invalid_result = valid & (~np.isfinite(proxy) | proxy.le(0))
    reason.loc[invalid_result] += "invalid_result;"
    proxy = proxy.where(valid & ~invalid_result)
    return pd.DataFrame({"market_cap_proxy": proxy,
                         "market_cap_invalid_reason": reason.str.rstrip(";")})


def _snapshot(config: Config, base: pd.DataFrame, source: str, mapping: dict,
              prefix: str) -> pd.DataFrame:
    path = config.source(source)
    if path is None or not path.is_file():
        if not config.optional_control_sources:
            raise ValueError(f"missing market-cap source: {source}")
        result = base.copy()
        result[f"{prefix}_date"] = pd.NaT
        for dest in mapping.values():
            result[dest] = np.nan
        return result
    available = columns(path)
    selected = [name for name in mapping if name in available]
    if len(selected) != len(mapping) and not config.optional_control_sources:
        raise ValueError(f"{source}: missing market-cap fields {sorted(set(mapping) - set(selected))}")
    frame = read_frame(path, ["date", "code"] + selected).rename(columns=mapping)
    for dest in mapping.values():
        if dest not in frame:
            frame[dest] = np.nan
    # Rows after the final cutoff are unnecessary, including entry-day closes.
    frame = frame[frame.date.le(base.cutoff.max()) & frame.code.isin(base.code.unique())]
    if getattr(config, "aiq_fill_enabled", False):
        from .aiq_fill import field_policy, filled_snapshot
        return filled_snapshot(base, frame, list(mapping.values()), prefix=prefix,
            calendar=read_frame(config.source("calendar"), ["date"]).date,
            fill_rules={dest: field_policy(original) for original, dest in mapping.items()})
    return asof_snapshot(base, frame, prefix)


def market_cap_snapshot(config: Config, base: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest same-day proxy inputs without filling missing events.

    Matching each feed backwards and accepting only equal source dates is
    equivalent to outer-joining daily feeds and then taking the latest record.
    An inner join would incorrectly skip a newer one-sided/missing update.
    """
    spec = config.market_cap
    raw_mapping = {spec["close_column"]: "market_cap_close",
                   spec["volume_column"]: "market_cap_volume"}
    turnover_mapping = {spec["turnover_column"]: "market_cap_turnover"}
    if spec["quotes_source"] == spec["turnover_source"]:
        result = _snapshot(config, base, spec["quotes_source"], raw_mapping | turnover_mapping,
                           "market_cap_quote")
        result["market_cap_turnover_date"] = result.market_cap_quote_date
    else:
        result = _snapshot(config, base, spec["quotes_source"], raw_mapping, "market_cap_quote")
        result = _snapshot(config, result, spec["turnover_source"], turnover_mapping, "market_cap_turnover")
    result["market_cap_date"] = result[["market_cap_quote_date", "market_cap_turnover_date"]].max(axis=1)
    same_date = (result.market_cap_quote_date.notna() & result.market_cap_turnover_date.notna()
                 & result.market_cap_quote_date.eq(result.market_cap_turnover_date))
    calculated = quote_market_cap(result.market_cap_close, result.market_cap_volume, result.market_cap_turnover)
    result["market_cap_proxy"] = calculated.market_cap_proxy.where(same_date)
    result["market_cap_invalid_reason"] = calculated.market_cap_invalid_reason
    result.loc[~same_date, "market_cap_invalid_reason"] = "source_date_mismatch"
    absent = result.market_cap_quote_date.isna() | result.market_cap_turnover_date.isna()
    result.loc[absent, "market_cap_invalid_reason"] = "missing_source_observation"
    return result
