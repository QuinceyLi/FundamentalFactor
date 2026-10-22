"""Point-in-time ST states from dated public records, with explicit unknowns.

These helpers never infer an earlier state from a later name.  The dated
exchange event describes the *new* name effective on that date; before the
first event a security remains unknown.  Missing vendor days terminate a
vendor state instead of silently extending it through the gap.
"""
from __future__ import annotations

import unicodedata

import pandas as pd

from .data import dates, require_unique


def st_from_names(names: pd.Series) -> pd.Series:
    """Recognise ST/*ST and historic S/G prefixes; absent names stay unknown."""
    normalized = names.map(lambda x: unicodedata.normalize("NFKC", str(x))
                           if pd.notna(x) else None).astype("string")
    normalized = normalized.str.replace(r"\s+", "", regex=True).str.upper()
    unknown = normalized.isna() | normalized.isin(["", "-", "--"])
    result = normalized.str.match(r"^(?:G)?(?:S)?\*?ST").astype("boolean")
    return result.mask(unknown, pd.NA)


def normalize_szse_names(raw: pd.DataFrame, observed_date: str | pd.Timestamp) -> pd.DataFrame:
    required = ["变更日期", "证券代码", "变更前简称", "变更后简称"]
    if any(name not in raw for name in required):
        raise ValueError("SZSE name-change response schema is incomplete")
    data = raw[raw["证券代码"].astype("string").str.fullmatch(r"[03]\d{5}", na=False)].copy()
    data = data.rename(columns={"变更日期": "date", "证券代码": "code",
                                "变更前简称": "name_before", "变更后简称": "name_after"})
    data = data[["date", "code", "name_before", "name_after"]]
    data["date"] = dates(data.date)
    data["code"] = data.code + ".SZ"
    observed = pd.Timestamp(observed_date).normalize()
    if data.empty or (data.date > observed).any():
        raise ValueError("Empty or future-dated SZSE name-change response")
    require_unique(data, ["date", "code"])
    data = data.sort_values(["code", "date"]).reset_index(drop=True)
    previous = data.groupby("code", sort=False).name_after.shift()
    # A broken chain is evidence of a missing event; do not publish an
    # apparently complete continuous state history in that case.
    mismatch = previous.notna() & previous.ne(data.name_before)
    if mismatch.any():
        raise ValueError(f"SZSE name-change chain is discontinuous: {int(mismatch.sum())} rows")
    data["st_in_name_before"] = st_from_names(data.name_before)
    data["st_in_name_after"] = st_from_names(data.name_after)
    data["source"] = "SZSE"
    data["observed_date"] = observed
    return data.sort_values(["date", "code"]).reset_index(drop=True)


def exchange_st_events(names: pd.DataFrame, valid_through: str | pd.Timestamp) -> pd.DataFrame:
    """Return as-of-ready states; append expiry rows to prevent future carry."""
    stop = pd.Timestamp(valid_through).normalize()
    names = names.copy()
    names["date"] = dates(names.date)
    require_unique(names, ["date", "code"])
    if (names.date > stop).any():
        raise ValueError("Name event exceeds the validated observation date")
    result = names[["date", "code", "name_after", "source"]].copy()
    result["name_after"] = result.name_after.astype("string")
    result["is_st"] = st_from_names(result.name_after)
    result["source_kind"] = "official_name_event"
    result["observed_date"] = stop
    result["valid_until"] = stop
    result["is_proxy"] = False
    last = result.sort_values("date").groupby("code", sort=False).tail(1).copy()
    last["date"] = stop + pd.Timedelta(days=1)
    last["is_st"] = pd.Series(pd.NA, index=last.index, dtype="boolean")
    last["name_after"] = pd.Series(pd.NA, index=last.index, dtype="string")
    last["source_kind"] = "coverage_expiry"
    result = pd.concat([result, last], ignore_index=True)
    result["is_st"] = result.is_st.astype("boolean")
    require_unique(result, ["date", "code"])
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def vendor_daily_st_events(daily: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """Compress vendor observations without carrying a state over missing days."""
    data = daily.copy()
    data["date"] = dates(data.date)
    require_unique(data, ["date", "code"])
    valid = data.is_st.dropna().isin([True, False, 0, 1])
    if not valid.all():
        raise ValueError("Vendor is_st must be boolean, 0/1, or missing")
    data["is_st"] = data.is_st.astype("boolean")
    sessions = pd.DatetimeIndex(dates(calendar).sort_values().drop_duplicates())
    if not data.date.isin(sessions).all():
        raise ValueError("Vendor observations contain a non-session date")
    blocks = []
    for code, block in data.groupby("code", sort=False):
        block = block.sort_values("date")
        position = sessions.get_indexer(block.date)
        successors = sessions[position[position + 1 < len(sessions)] + 1]
        if len(position) and position[-1] == len(sessions) - 1:
            # Beyond the supplied calendar's coverage no observation is known.
            successors = successors.append(pd.DatetimeIndex([sessions[-1] + pd.Timedelta(days=1)]))
        missing = successors.difference(pd.DatetimeIndex(block.date))
        extension = pd.DataFrame({"date": missing, "code": code,
                                  "is_st": pd.Series([pd.NA] * len(missing), dtype="boolean"),
                                  "source": "BaoStock", "source_kind": "missing_vendor_day"})
        block = block.assign(source_kind="third_party_daily")
        block = pd.concat([block, extension], ignore_index=True).sort_values("date")
        # NA is a genuine state here.  Keep transitions into and out of it.
        values = block.is_st.astype("Int8").fillna(-1)
        blocks.append(block.loc[values.ne(values.shift()).fillna(True)])
    if not blocks:
        return pd.DataFrame(columns=["date", "code", "is_st", "source", "source_kind"])
    return pd.concat(blocks, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
