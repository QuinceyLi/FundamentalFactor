"""Prepare traceable local reference inputs; source files are always read-only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from .data import asof_snapshot, dates, fingerprint, read_frame, require_unique, write_json
from .proxy_inputs import trade_status_proxy


def merge_st_sources(aiq: pd.DataFrame, official: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer verified AIQ states; a contradictory official event takes precedence.

    Union event days before matching. NA is a state transition, never ffilled
    away; neither source is inferred before its first event or after expiry.
    """
    for frame in (aiq, official):
        require_unique(frame, ["date", "code"])
    left = pd.concat([aiq[["date", "code"]], official[["date", "code"]]], ignore_index=True).drop_duplicates()
    left["cutoff"] = left.date
    a = aiq.rename(columns={"is_st": "aiq_is_st"}).copy()
    if "st_source" not in a:
        a["st_source"] = "AIQ"
    if "st_status_reason" not in a:
        a["st_status_reason"] = "aiq_candidate"
    a = a.rename(columns={"st_source": "aiq_source", "st_status_reason": "aiq_reason"})
    fields = ["date", "code", "aiq_is_st", "aiq_source", "aiq_reason"]
    merged = asof_snapshot(left, a[fields], "aiq")
    merged = asof_snapshot(merged, official[["date", "code", "is_st"]].rename(columns={"is_st": "official_is_st"}), "official")
    aiq_known, official_known = merged.aiq_is_st.notna(), merged.official_is_st.notna()
    conflict = (aiq_known & official_known & merged.aiq_is_st.ne(merged.official_is_st)).fillna(False)
    use_official = official_known & (~aiq_known | conflict)
    merged["is_st"] = merged.aiq_is_st.astype("boolean").where(~use_official, merged.official_is_st.astype("boolean"))
    merged["st_source"] = merged.aiq_source.fillna("unknown")
    merged.loc[use_official, "st_source"] = "SZSE_official_name_event"
    merged["st_status_reason"] = merged.aiq_reason.fillna("no_aiq_or_official_evidence")
    merged.loc[use_official, "st_status_reason"] = "official_fallback"
    merged.loc[conflict, "st_status_reason"] = "official_event_overrides_aiq_conflict"
    merged["st_known"] = merged.is_st.notna()
    merged["st_conflict"] = conflict
    conflicts = merged.loc[conflict].copy()
    columns = ["date", "code", "is_st", "st_known", "st_source", "st_status_reason", "st_conflict", "aiq_date", "official_date"]
    output = merged[columns].sort_values(["date", "code"]).reset_index(drop=True)
    require_unique(output, ["date", "code"])
    return output, conflicts


def prepare_calendar(aiq_source: Path, fallback: Path, output: Path) -> dict:
    aiq = read_frame(aiq_source, ["date"]).drop_duplicates().sort_values("date")
    old = read_frame(fallback, ["date"]).drop_duplicates().sort_values("date")
    # Source coverage is explicit. Earlier official calendar history remains
    # useful for listing age, but no missing day inside AIQ is silently added.
    extension = old.loc[old.date.lt(aiq.date.min()) | old.date.gt(aiq.date.max())]
    result = pd.concat([aiq.assign(calendar_source="AIQ"), extension.assign(calendar_source="public_calendar_fallback")]).sort_values("date")
    require_unique(result, ["date"])
    differences = old.loc[old.date.between(aiq.date.min(), aiq.date.max()) & ~old.date.isin(aiq.date)]
    differences.to_csv(output.with_suffix(".differences.csv"), index=False)
    result.to_parquet(output, index=False)
    return {"source": fingerprint(aiq_source), "fallback": fingerprint(fallback), "output": fingerprint(output),
            "rows": len(result), "start": result.date.min(), "end": result.date.max(),
            "fallback_rows": len(extension), "existing_days_missing_inside_aiq_coverage": len(differences)}


def prepare_stocks(aiq_source: Path, delist_source: Path, fallback: Path, output: Path) -> dict:
    aiq = pd.read_csv(aiq_source, dtype={"code": "string"}).rename(columns={"ipo_date": "listing_date", "delist_date": "delisting_date"})
    old = read_frame(fallback, ["code", "listing_date", "delisting_date"])
    for col in ("listing_date", "delisting_date"):
        aiq[col] = dates(aiq[col], nullable=col == "delisting_date")
    aiq = aiq[aiq.code.str.fullmatch(r"(?:6\d{5}\.SH|[03]\d{5}\.SZ)", na=False)].copy()
    require_unique(aiq, ["code"])
    require_unique(old, ["code"])
    delists = pd.read_csv(delist_source, dtype={"code": "string"})
    delists["delisting_date"] = dates(delists.delisting_date)
    require_unique(delists, ["code"])
    # The separate table states S_INFO_DELISTDATE explicitly; do not confuse
    # it with last_trading_date_exchange or last_trading_date_stock.
    aiq = aiq.merge(delists[["code", "delisting_date"]].rename(columns={"delisting_date": "explicit_delisting_date"}), on="code", how="left", validate="one_to_one")
    aiq["delisting_date"] = aiq.explicit_delisting_date.combine_first(aiq.delisting_date)
    aiq = aiq.drop(columns="explicit_delisting_date")
    compared = aiq.merge(old, on="code", suffixes=("_aiq", "_existing"))
    conflict = (compared.listing_date_aiq.ne(compared.listing_date_existing) |
        compared.delisting_date_aiq.fillna(pd.Timestamp.max.normalize()).ne(compared.delisting_date_existing.fillna(pd.Timestamp.max.normalize())))
    compared.loc[conflict].to_csv(output.with_suffix(".differences.csv"), index=False)
    aiq["stock_source"] = "AIQ_security_master_and_explicit_delisting_date"
    supplemental = old.loc[~old.code.isin(aiq.code)].assign(stock_source="public_exchange_master_fallback")
    result = pd.concat([aiq, supplemental], ignore_index=True)
    if (result.delisting_date.notna() & result.delisting_date.lt(result.listing_date)).any():
        raise ValueError("Invalid security lifecycle in combined master")
    result["is_a_share"] = True
    result.sort_values("code").to_parquet(output, index=False)
    return {"source": fingerprint(aiq_source), "delisting_source": fingerprint(delist_source), "fallback": fingerprint(fallback),
        "output": fingerprint(output), "rows": len(result), "aiq_rows": len(aiq), "fallback_rows": len(supplemental),
        "listing_date_disagreements": int(compared.listing_date_aiq.ne(compared.listing_date_existing).sum()),
        "disagreement_rows": int(conflict.sum()), "policy": "AIQ lifecycle preferred, explicit delisting_date preferred to last trading dates; conflicts disclosed; public master supplies absent securities"}


def prepare_trade_status(aiq_quotes: Path, stoptrade_source: Path, fallback: Path,
                         st_history: Path, stocks_path: Path, calendar_path: Path,
                         output: Path, start, end) -> dict:
    # Deliberately project only raw fields. Known bad AIQ adjusted fields are
    # never read or copied into the repaired return-price source.
    quotes = feather.read_table(aiq_quotes, columns=["date", "code", "open", "close", "volume", "last"]).to_pandas()
    quotes["date"] = dates(quotes.date)
    quotes["code"] = quotes.code.astype("string")
    stocks = read_frame(stocks_path, ["code", "listing_date"])
    quotes = quotes.loc[quotes.date.between(pd.Timestamp(start), pd.Timestamp(end)) & quotes.code.isin(stocks.code)].reset_index(drop=True)
    require_unique(quotes, ["date", "code"])
    st = read_frame(st_history, ["date", "code", "is_st"], nullable_booleans=True)
    daily = asof_snapshot(quotes[["date", "code"]].rename(columns={"date": "cutoff"}), st, "st")
    daily = daily.rename(columns={"cutoff": "date"})
    calendar = read_frame(calendar_path, ["date"]).date
    status = trade_status_proxy(quotes, daily, stocks, calendar)
    status["status_source"] = "AIQ_raw_quote_volume_close_proxy"
    old = read_frame(fallback, nullable_booleans=True)
    old = old.loc[old.date.between(pd.Timestamp(start), pd.Timestamp(end)) & old.code.isin(stocks.code)]
    require_unique(old, ["date", "code"])
    compared = status[["date", "code", "suspended"]].merge(old[["date", "code", "suspended"]], on=["date", "code"], suffixes=("_aiq", "_existing"))
    different = (compared.suspended_aiq.notna() & compared.suspended_existing.notna() & compared.suspended_aiq.ne(compared.suspended_existing)).fillna(False)
    compared.loc[different].to_csv(output.with_suffix(".disagreements.csv"), index=False)
    aiq_keys = pd.MultiIndex.from_frame(status[["date", "code"]])
    supplemental = old.loc[~pd.MultiIndex.from_frame(old[["date", "code"]]).isin(aiq_keys)].copy()
    supplemental["status_source"] = "existing_raw_quote_proxy_fallback"
    result = pd.concat([status, supplemental], ignore_index=True).sort_values(["date", "code"])
    require_unique(result, ["date", "code"])
    result.to_parquet(output, index=False)
    events = pd.read_csv(stoptrade_source, usecols=["date", "code"], dtype={"code": "string"})
    valid = events.code.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)", na=False)
    invalid_events = events.loc[~valid].copy()
    invalid_events.to_csv(output.with_suffix(".event_quarantine.csv"), index=False)
    events = events.loc[valid].copy()
    events["date"] = dates(events.date)
    events = events.drop_duplicates(["date", "code"])
    matched = events.merge(status[["date", "code", "suspended"]], on=["date", "code"], how="inner")
    nonzero = matched.loc[matched.suspended.eq(False).fillna(False)]
    nonzero.head(500).to_csv(output.with_suffix(".suspension_event_examples.csv"), index=False)
    return {"source": fingerprint(aiq_quotes), "fallback": fingerprint(fallback), "event_source": fingerprint(stoptrade_source),
        "output": fingerprint(output), "rows": len(result), "aiq_rows": len(status), "fallback_rows": len(supplemental),
        "raw_proxy_disagreements": int(different.sum()), "suspension_event_matches": len(matched),
        "noncanonical_suspension_events_quarantined": len(invalid_events),
        "suspension_events_with_nonzero_volume_close": len(nonzero),
        "limitations": ["Complete-day volume==0 or close==0 is a research proxy, not opening-auction knowledge.",
            "Sparse stoptrade start events do not prove full-day suspension; nonzero-volume events are disclosed, not forced to suspended.",
            "Limits remain board/ST/ex-rights-reference opening proxies; sparse old close-limit events are not substituted for opening restrictions.",
            "AIQ adjusted prices are not loaded, repaired return quotes remain unchanged."]}
