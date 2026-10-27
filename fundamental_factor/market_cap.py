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
