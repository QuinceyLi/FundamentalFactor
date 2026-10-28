"""AIQ daily matrix fill followed by explicitly labelled GF trading proxies.

Only raw price/volume fields are read. Filling never edits a source file or
turns an unavailable actual exit value into a settlement. Small stock batches
share the full source history, preserving donors across calendar years.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as parquet

from .aiq_fill import filled_snapshot
from .data import asof_snapshot, dates, fingerprint, read_frame, require_unique, write_json
from .proxy_inputs import trade_status_proxy


RAW_FIELDS = ("open", "close", "volume", "last")


def prepare_aiq_matrix_status(quote_source: str | Path, calendar_source: str | Path,
                              stocks_source: str | Path, st_source: str | Path,
                              output: str | Path, *, start=None, end=None,
                              stock_batch_size: int = 128) -> dict:
    """Export lifecycle-limited daily flags with original/fill/donor columns.

    ``open`` stays missing; ``close``/``last`` forward-fill; ``volume`` becomes
    zero after its first actual value. Zero, negative and infinite source
    values are never imputed away. GF's zero-volume/zero-close rule is applied
    afterwards; invalid remaining raw inputs produce an unknown suspension
    flag. ST events retain explicit unknown transitions and are never filled.
    """
    sources = {name: Path(value).resolve() for name, value in {
        "raw_quotes": quote_source, "calendar": calendar_source,
        "stocks": stocks_source, "st_history": st_source}.items()}
    output = Path(output).resolve()
    if output.suffix != ".parquet" or output in sources.values():
        raise ValueError("Matrix status needs a separate .parquet artifact")
    if any(output.is_relative_to(path) for path in (Path("/data/AIQ"), Path("/data/mongo"))):
        raise ValueError("AIQ and Mongo source directories are read-only")
    if isinstance(stock_batch_size, bool) or not isinstance(stock_batch_size, int) or stock_batch_size < 1:
        raise ValueError("stock_batch_size must be a positive integer")
    original = {name: fingerprint(path) for name, path in sources.items()}
    calendar = read_frame(sources["calendar"], ["date"]).date.sort_values().reset_index(drop=True)
    if calendar.empty or calendar.duplicated().any():
        raise ValueError("Calendar must contain distinct trading dates")
    stocks = read_frame(sources["stocks"], ["code", "listing_date", "delisting_date"])
    require_unique(stocks, ["code"])
    if (stocks.delisting_date.notna() & stocks.delisting_date.lt(stocks.listing_date)).any():
        raise ValueError("Invalid security lifecycle")
    st = read_frame(sources["st_history"], ["date", "code", "is_st"], nullable_booleans=True)
    require_unique(st, ["date", "code"])
    quote_path = sources["raw_quotes"]
    columns = ["date", "code", *RAW_FIELDS]
    quotes = (feather.read_table(quote_path, columns=columns, memory_map=True).to_pandas()
              if quote_path.suffix == ".feather" else read_frame(quote_path, columns))
    quotes["date"] = dates(quotes.date).astype("datetime64[ns]")
    quotes["code"] = quotes.code.astype("string")
    quotes = quotes.loc[quotes.code.isin(stocks.code) & quotes.date.isin(calendar)].copy()
    if quotes.empty:
        raise ValueError("No quoted security matches the supplied lifecycle/calendar")
    for field in RAW_FIELDS:
        quotes[field] = pd.to_numeric(quotes[field], errors="raise").astype(float)
    require_unique(quotes, ["date", "code"])
    start = quotes.date.min() if start is None else pd.Timestamp(start).normalize()
    end = quotes.date.max() if end is None else pd.Timestamp(end).normalize()
    sessions = calendar.loc[calendar.between(start, end)].reset_index(drop=True)
    if sessions.empty or end < start:
        raise ValueError("Requested status interval has no trading dates")
    # A donor before listing must not leak into a later live-stock gap.
    lifecycle = quotes[["code"]].merge(stocks, on="code", how="left", validate="many_to_one")
    live_quote = (quotes.date.to_numpy() >= lifecycle.listing_date.to_numpy()) & (
        lifecycle.delisting_date.isna().to_numpy() | (quotes.date.to_numpy() < lifecycle.delisting_date.to_numpy()))
    source_outside_lifecycle = int((~live_quote).sum())
    quotes = quotes.loc[live_quote & quotes.date.le(end).to_numpy()].reset_index(drop=True)
    del lifecycle, live_quote
    positions = quotes.groupby("code", sort=False).indices
    stock_codes = sorted(stocks.code.tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    writer = None
    totals, years = Counter(), Counter()
    date_start, date_end = None, None
    try:
        for offset in range(0, len(stock_codes), stock_batch_size):
            batch_codes = stock_codes[offset:offset + stock_batch_size]
            batch_stocks = stocks.loc[stocks.code.isin(batch_codes)]
            indices = [positions[code] for code in batch_codes if code in positions]
            source = quotes.iloc[np.concatenate(indices)] if indices else quotes.iloc[:0]
            grid = pd.MultiIndex.from_product([sessions, batch_codes], names=["cutoff", "code"]).to_frame(index=False)
            grid = grid.merge(batch_stocks, on="code", how="left", validate="many_to_one")
            live = grid.cutoff.ge(grid.listing_date) & (grid.delisting_date.isna() | grid.cutoff.lt(grid.delisting_date))
            grid = grid.loc[live, ["cutoff", "code"]].reset_index(drop=True)
            if grid.empty:
                continue
            filled = filled_snapshot(grid, source, RAW_FIELDS, prefix="quote", calendar=calendar)
            history = asof_snapshot(grid, st.loc[st.code.isin(batch_codes)], "st").rename(columns={"cutoff": "date"})
            filled = filled.rename(columns={"cutoff": "date"}).reset_index(drop=True)
            status = trade_status_proxy(filled[["date", "code", *RAW_FIELDS]], history,
                                        batch_stocks[["code", "listing_date"]], calendar)
            # Positional joining is safe here: trade_status_proxy preserves
            # exactly the filled keys/order. Guard that invariant explicitly.
            if not status[["date", "code"]].equals(filled[["date", "code"]]):
                raise AssertionError("Status proxy changed the quote grid")
            result = pd.concat([status, filled.drop(columns=["date", "code"])], axis=1)
            result["status_source"] = "AIQ_raw_daily_matrix_fill_GF_proxy"
            result["missing_values_method"] = "aiq_matrix"
            totals["rows"] += len(result)
            totals["codes"] += result.code.nunique()
            totals["suspended_rows"] += int(result.suspended.fillna(False).sum())
            totals["suspension_unknown_rows"] += int(result.suspended.isna().sum())
            totals["st_unknown_rows"] += int(result.st_unknown.sum())
            totals["limit_up_unknown_rows"] += int(result.limit_up.isna().sum())
            totals["limit_down_unknown_rows"] += int(result.limit_down.isna().sum())
            totals["no_raw_quote_key_rows"] += int(result.quote_record_date.ne(result.date).sum())
            for field in RAW_FIELDS:
                totals[f"{field}_original_missing_rows"] += int(result[f"{field}_original_missing"].sum())
                totals[f"{field}_filled_rows"] += int(result[f"{field}_filled"].sum())
            years.update(result.date.dt.year.value_counts().to_dict())
            date_start = result.date.min() if date_start is None else min(date_start, result.date.min())
            date_end = result.date.max() if date_end is None else max(date_end, result.date.max())
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = parquet.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
        if writer is None:
            raise ValueError("No live securities in the requested status interval")
        writer.close()
        writer = None
        for name, path in sources.items():
            if fingerprint(path) != original[name]:
                raise RuntimeError(f"Source changed during matrix status preparation: {name}")
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()
    summary = {"adapter": "aiq_matrix_trade_status_v1", "missing_values": {"method": "aiq_matrix"},
        "sources": original, "output": fingerprint(output), "counts": dict(totals),
        "rows": totals["rows"], "codes": totals["codes"], "aiq_rows": totals["rows"], "fallback_rows": 0,
        "start": date_start, "end": date_end, "annual_rows": dict(sorted(years.items())),
        "source_outside_lifecycle_rows_ignored": source_outside_lifecycle,
        "stock_batch_size": stock_batch_size,
        "field_rules": {"open": "keep", "close": "ffill", "volume": "zero_after_first_observation", "last": "ffill"},
        "source_reference": "/data/AIQ/Tools/datahub.py:FILL_RULES/_apply_fill; GF volume==0 or close==0",
        "limitations": [
            "Generated zero volume is a proxy assumption; original missing values and fill anchors remain exported.",
            "No backfill; first-observation gaps remain unknown; all donors and targets respect listing and delisting dates.",
            "All-day inputs do not establish opening-auction availability or executable trading limits.",
            "ST uses dated verified/proxy events with explicit unknown expiry, not generic matrix filling.",
            "Only raw quote fields are read; repaired adjusted return prices and all original files are unchanged.",
            "Rows are stored in stock batches; consumers must join by date/code, not assume global row order."]}
    write_json(output.with_suffix(".manifest.json"), summary)
    return summary
