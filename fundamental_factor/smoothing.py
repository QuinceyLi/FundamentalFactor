"""Optional causal smoothing of already computed factor snapshots.

EMA is evaluated once per rebalance period, separately for each stock, before
cross-sectional transforms. It may reduce signal noise and trading, but is not
a turnover constraint and cannot guarantee a lower portfolio turnover rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SMOOTHING_COLUMNS = ["smoothed_value", "smoothing_observations", "smoothing_reset"]


def smoothing_settings(settings: dict | None = None) -> dict:
    """Validate and resolve defaults without mutating a frozen configuration."""
    if settings is None:
        settings = {}
    if not isinstance(settings, dict) or set(settings) - {"method", "span", "min_periods"}:
        raise ValueError("smoothing only supports method, span and min_periods")
    result = {"method": "none", "span": 4, "min_periods": 1} | settings
    if not isinstance(result["method"], str) or result["method"] not in {"none", "ema"}:
        raise ValueError("smoothing.method must be none or ema")
    for key in ("span", "min_periods"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"smoothing.{key} must be a positive integer")
    if result["min_periods"] > result["span"]:
        raise ValueError("smoothing.min_periods must not exceed span")
    return result


def smooth_factor(frame: pd.DataFrame, periods: pd.DataFrame, settings: dict | None = None) -> pd.DataFrame:
    """Append an EMA while retaining original values, index and row order.

    ``frame`` contains one factor, keyed by entry/code, including explicit
    missing factor values. ``periods.entry`` is the complete rebalance grid.
    For consecutive valid observations, s[t] = alpha*x[t] + (1-alpha)*s[t-1]
    with alpha=2/(span+1), equivalent to adjust=False. A new stock, absent grid
    period, or nonfinite input resets the state; no old value can fill the
    current missing observation. The next valid segment starts at its first
    observation. min_periods counts consecutive valid observations in a segment.

    The first requested period starts a new history: no implicit warm-up data
    outside the frozen research grid is read. Changing the research start can
    therefore change early EMA values. The default method=none is a pure copy.
    """
    spec = smoothing_settings(settings)
    result = frame.copy()
    if spec["method"] == "none":
        return result
    required = {"entry", "code", "value"}
    if missing := required - set(frame):
        raise ValueError(f"smoothing missing fields: {sorted(missing)}")
    if "entry" not in periods:
        raise ValueError("smoothing requires a complete periods.entry grid")
    if set(SMOOTHING_COLUMNS) & set(frame):
        raise ValueError("factor snapshots already contain smoothing output columns")
    grid = pd.DatetimeIndex(pd.to_datetime(periods.entry, errors="raise"))
    if grid.hasnans or grid.has_duplicates or grid.tz is not None:
        raise ValueError("smoothing period entries must be unique non-null timezone-naive timestamps")
    grid = grid.sort_values()
    entries = pd.DatetimeIndex(pd.to_datetime(frame.entry, errors="raise"))
    codes = frame.code.astype("string")
    if entries.hasnans or entries.tz is not None or codes.isna().any() or codes.str.strip().eq("").any():
        raise ValueError("smoothing keys must be non-null timezone-naive entries and nonempty codes")
    keys = pd.DataFrame({"entry": entries, "code": codes.to_numpy()})
    if keys.duplicated(["entry", "code"]).any():
        raise ValueError("duplicate factor smoothing entry/code keys")
    locations = grid.get_indexer(entries)
    if (locations < 0).any():
        raise ValueError("factor entries fall outside the complete smoothing period grid")
    values = pd.to_numeric(frame.value, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    groups, _ = pd.factorize(codes, sort=False)
    order = np.lexsort((locations, groups))
    smoothed = np.full(len(frame), np.nan)
    observations = np.zeros(len(frame), dtype=np.int64)
    resets = np.zeros(len(frame), dtype=bool)
    alpha = 2. / (spec["span"] + 1.)
    last_group, last_location, state, count = -1, -2, np.nan, 0
    for position in order:
        group, location, value = groups[position], locations[position], values[position]
        contiguous = group == last_group and location == last_location + 1
        if not contiguous or not np.isfinite(state):
            state, count = np.nan, 0
        if np.isfinite(value):
            resets[position] = count == 0
            if count == 0 or alpha == 1.:
                state = value
            else:
                # A convex weighted sum avoids overflow in (value - state)
                # for finite inputs of opposite signs and large magnitudes.
                with np.errstate(over="ignore", invalid="ignore"):
                    state = alpha * value + (1. - alpha) * state
            count += 1
            observations[position] = count
            if count >= spec["min_periods"] and np.isfinite(state):
                smoothed[position] = state
        else:
            state, count = np.nan, 0
        last_group, last_location = group, location
    result["smoothed_value"] = smoothed
    result["smoothing_observations"] = observations
    result["smoothing_reset"] = resets
    return result
