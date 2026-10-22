"""Explicit research proxies; no inferred flag is labelled an exchange record."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .data import asof_snapshot, fingerprint, read_frame, require_unique, write_json
from .gf_adapter import status_from_quotes


def compact_date(values):
    return (values.dt.year * 10000 + values.dt.month * 100 + values.dt.day).astype("Int64")


def history_from_sources(keys, classes, st_events):
    """Use only historical records; a future name change cannot clear past ST."""
    require_unique(classes, ["date", "code"])
    require_unique(st_events, ["date", "code"])
    native = classes[["date", "code", "class"]].copy()
    valid = native["class"].between(1, 35) & native["class"].mod(1).eq(0)
    native["industry"] = native["class"].where(valid).astype("Int64").astype("string").radd("class_")
    base = keys[["date", "code"]].drop_duplicates().rename(columns={"date": "cutoff"})
    history = asof_snapshot(base, native, "industry")
    history = asof_snapshot(history, st_events[["date", "code", "is_st"]], "st")
    history = history.rename(columns={"cutoff": "date"})
    history["is_st"] = history.is_st.astype("boolean")
    history["st_unknown"] = history.is_st.isna()
    history["industry_method"] = "historical_native_class_proxy"
    return history.sort_values(["date", "code"]).reset_index(drop=True)


def trade_status_proxy(quotes, history, stocks, calendar):
    """GF zero-volume rule plus directional opening-price limit approximations.

    Use vendor ex-rights reference `last`, not yesterday's raw close. Five first
    listing sessions are deliberately unknown; historical special exemptions
    and restoration rules are not certified. Status is diagnostic, not tradability.
    """
    quotes = quotes.reset_index(drop=True)
    flags = status_from_quotes(quotes)
    data = quotes.merge(history[["date", "code", "is_st"]], on=["date", "code"], how="left",
                        validate="one_to_one").merge(stocks[["code", "listing_date"]], on="code",
                        how="left", validate="many_to_one")
    board20 = data.code.str.startswith("688") | (data.code.str.startswith(("300", "301")) &
                                                              data.date.ge("2020-08-24"))
    board30 = data.code.str.endswith(".BJ")
    rate = pd.Series(.10, index=data.index)
    # Both exchanges' 2026 rules changed main-board ST from 5% to 10% on July 6.
    rate.loc[data.is_st.fillna(False).astype(bool) & data.date.lt("2026-07-06")] = .05
    rate.loc[board20] = .20
    rate.loc[board30] = .30
    days = np.sort(pd.to_datetime(calendar).unique())
    age = np.searchsorted(days, data.date.to_numpy(), side="right") - np.searchsorted(
        days, data.listing_date.to_numpy(), side="left")
    ref = pd.to_numeric(data["last"], errors="coerce")
    valid = np.isfinite(ref) & ref.gt(0) & np.isfinite(data.open) & data.open.gt(0)
    valid &= data.listing_date.notna() & (age > 5)
    # Guard binary floating error at an exact half cent (4.30*1.05=4.515).
    upper = np.floor(ref * (1 + rate) * 100 + .5 + 1e-8) / 100
    lower = np.floor(ref * (1 - rate) * 100 + .5 + 1e-8) / 100
    flags["limit_up"] = data.open.ge(upper).astype("boolean").where(valid)
    flags["limit_down"] = data.open.le(lower).astype("boolean").where(valid)
    flags["limit_pct_proxy"] = rate.where(valid)
    flags["limit_method"] = "board_st_reference_open_proxy"
    flags["st_unknown"] = data.is_st.isna()
    flags["ipo_or_reference_unknown"] = ~valid
    return flags


def prepare_proxy_inputs(output: Path, quote_path: Path, class_path: Path, st_path: Path,
                         stocks_path: Path, calendar_path: Path):
    output.mkdir(parents=True, exist_ok=True)
    quotes = read_frame(quote_path, ["date", "code", "open", "close", "volume", "last"])
    quotes = quotes[quotes.code.str.endswith((".SH", ".SZ"))].reset_index(drop=True)
    classes = read_frame(class_path, ["date", "code", "class"])
    st = read_frame(st_path, ["date", "code", "is_st"], nullable_booleans=True)
    stocks = read_frame(stocks_path, ["code", "listing_date"])
    calendar = read_frame(calendar_path, ["date"]).date
    history = history_from_sources(quotes, classes, st)
    status = trade_status_proxy(quotes, history, stocks, calendar)
    datasets = {"stock_history_proxy.feather": history, "stock_trade_status_proxy.feather": status}
    artifacts = {}
    for name, frame in datasets.items():
        frame = frame.copy()
        # Match Mongo date/code convention. Nullable provenance dates remain Int64.
        for field in ("date", "industry_date", "st_date"):
            if field in frame:
                frame[field] = compact_date(frame[field])
        frame.to_feather(output / name)
        artifacts[name] = fingerprint(output / name)
    manifest = {"sources": {str(p): fingerprint(p) for p in
                (quote_path, class_path, st_path, stocks_path, calendar_path)},
        "rows": len(history), "codes": history.code.nunique(),
        "start": history.date.min(), "end": history.date.max(),
        "st_known_rows": int(history.is_st.notna().sum()), "st_true_rows": int(history.is_st.sum()),
        "st_unknown_fraction": float(history.is_st.isna().mean()),
        "industry_missing_rows": int(history.industry.isna().sum()),
        "suspension_proxy_rows": int(status.suspended.sum()),
        "limit_up_proxy_rows": int(status.limit_up.sum()), "limit_down_proxy_rows": int(status.limit_down.sum()),
        "limit_unknown_rows": int(status.limit_up.isna().sum()), "artifacts": artifacts,
        "limitations": ["ST absent before first verified record stays unknown; never backfilled from today's name.",
            "Industry uses native historical class 1..35. Class 35 mixes industries; no claim of an official 30-sector classification.",
            "Volume/close proxy uses complete-day observations, not information known at the opening auction.",
            "Limit proxies use vendor ex-rights last, ST when known, 10/20/30 percent board rules, ChiNext 2020-08-24 and main-board ST 2026-07-06 boundaries.",
            "Unknown ST uses ordinary board rate solely for limit approximation; IPO first five sessions unknown; special resume/exemption rules incomplete.",
            "Only SH/SZ A shares. Corporate action and actual terminal settlement completeness not asserted."]}
    write_json(output / "proxy_inputs_manifest.json", manifest)
    return manifest


def refresh_trade_status(output, quote_path, stocks_path, calendar_path):
    """Recompute limits after a rule update without repeating historical joins."""
    import json
    history = read_frame(output / "stock_history_proxy.feather", ["date", "code", "is_st"], nullable_booleans=True)
    quotes = read_frame(quote_path, ["date", "code", "open", "close", "volume", "last"])
    quotes = quotes[quotes.code.str.endswith((".SH", ".SZ"))].reset_index(drop=True)
    status = trade_status_proxy(quotes, history, read_frame(stocks_path, ["code", "listing_date"]),
                                read_frame(calendar_path, ["date"]).date)
    status["date"] = compact_date(status.date)
    status.to_feather(output / "stock_trade_status_proxy.feather")
    manifest = json.loads((output / "proxy_inputs_manifest.json").read_text())
    manifest.update(limit_up_proxy_rows=int(status.limit_up.sum()), limit_down_proxy_rows=int(status.limit_down.sum()),
                    limit_unknown_rows=int(status.limit_up.isna().sum()))
    manifest["artifacts"]["stock_trade_status_proxy.feather"] = fingerprint(output / "stock_trade_status_proxy.feather")
    manifest["producer"] = fingerprint(Path(__file__).resolve())
    manifest["trading_rules"] = json.loads((output / "trading_rules.json").read_text()) if (output / "trading_rules.json").exists() else {}
    manifest["limitations"] = [x for x in manifest["limitations"] if not x.startswith("Limit proxies")]
    manifest["limitations"].append("Limit proxies use last reference; ChiNext regime 2020-08-24 and main-board ST regime 2026-07-06. Special exemptions remain incomplete.")
    write_json(output / "proxy_inputs_manifest.json", manifest)
    return manifest
