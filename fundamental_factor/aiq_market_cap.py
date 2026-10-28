"""Read-only AIQ market-cap adapter and local, same-day corroboration.

AIQ ``circulated_market_value`` is delivered in ten-thousand yuan. Share
events are in ten-thousand shares; only raw (unadjusted) closes can be used
with them. The adapter never repairs or writes a price feed. Disagreement
with a share history is normally diagnostic, since the history can lag.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .data import fingerprint, write_json


_KEYS = ["date", "code"]
_CODE_PATTERN = r"(?:6\d{5}\.SH|[03]\d{5}\.SZ|[489]\d{5}\.BJ)"
_OUTPUT_SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")), ("code", pa.string()),
    ("float_mcap", pa.float64()), ("market_cap_source", pa.string()),
    ("market_cap_status", pa.string()), ("market_cap_crosscheck_status", pa.string()),
    ("market_cap_invalid_reason", pa.string()),
    ("raw_mcap_wan", pa.float64()), ("share_implied_mcap", pa.float64()),
    ("proxy_mcap", pa.float64()), ("share_date", pa.timestamp("ns")),
    ("share_relative_error", pa.float64()), ("proxy_relative_error", pa.float64()),
])
_QUARANTINE_SCHEMA = pa.schema([
    ("origin", pa.string()), ("date", pa.timestamp("ns")),
    ("raw_date", pa.string()), ("code", pa.string()),
    ("raw_value", pa.float64()), ("reason", pa.string()),
    ("share_implied_mcap", pa.float64()), ("proxy_mcap", pa.float64()),
])


class _Writer:
    def __init__(self, path: Path, schema: pa.Schema):
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")
        self.schema = schema

    def append(self, frame: pd.DataFrame) -> None:
        if not frame.empty:
            frame = frame.reindex(columns=self.schema.names)
            self.writer.write_table(pa.Table.from_pandas(frame, schema=self.schema,
                                                        preserve_index=False, safe=True))

    def close(self) -> None:
        self.writer.close()


def _column_names(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return pq.read_schema(path).names
    if path.suffix.lower() == ".feather":
        with pa.memory_map(str(path), "r") as handle:
            return ipc.open_file(handle).schema.names
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="gb18030", nrows=0).columns.tolist()
    raise ValueError(f"Unsupported market-cap input: {path.suffix}")


def _batches(path: Path, selected: list[str], batch_size: int):
    """Project columns before conversion to pandas; never load an entire feed."""
    if path.suffix.lower() == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=selected):
            yield batch.to_pandas()
    elif path.suffix.lower() == ".feather":
        with pa.memory_map(str(path), "r") as handle:
            reader = ipc.open_file(handle)
            for number in range(reader.num_record_batches):
                batch = reader.get_batch(number).select(selected)
                for offset in range(0, batch.num_rows, batch_size):
                    yield batch.slice(offset, batch_size).to_pandas()
    elif path.suffix.lower() == ".csv":
        yield from pd.read_csv(path, usecols=selected, encoding="gb18030",
                               dtype={"code": "string"}, chunksize=batch_size)
    else:
        raise ValueError(f"Unsupported market-cap input: {path.suffix}")


def _dates(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        result = pd.to_datetime(values, errors="coerce")
        if getattr(result.dt, "tz", None) is not None:
            raise ValueError("AIQ daily dates must be timezone-naive")
        return result.where(result.eq(result.dt.normalize()))
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    result.loc[compact] = pd.to_datetime(text[compact], format="%Y%m%d", errors="coerce")
    result.loc[~compact] = pd.to_datetime(text[~compact], errors="coerce")
    return result.where(result.eq(result.dt.normalize()))


def _positive(values: pd.Series) -> pd.Series:
    return pd.Series(np.isfinite(values), index=values.index) & values.gt(0)


def _quarantine(frame: pd.DataFrame, origin: str, reason: str | pd.Series,
                raw_column: str, writer: _Writer) -> None:
    if frame.empty:
        return
    rows = frame.reindex(columns=["date", "code", "raw_date", "share_implied_mcap", "proxy_mcap"]).copy()
    rows["origin"] = origin
    rows["raw_value"] = pd.to_numeric(frame[raw_column], errors="coerce")
    rows["reason"] = reason
    writer.append(rows)


def _partition(path: Path, required: list[str], optional: list[str], folder: Path,
               calendar: set, batch_size: int, quarantine: _Writer, origin: str) -> dict:
    available = _column_names(path)
    missing = sorted(set(required) - set(available))
    if missing:
        raise ValueError(f"{origin}: missing fields {missing}")
    selected = required + [col for col in optional if col in available and col not in required]
    numeric = [col for col in selected if col not in _KEYS]
    stats = Counter()
    writers = {}
    first_date, last_date = None, None
    try:
        for batch in _batches(path, selected, batch_size):
            stats["source_rows"] += len(batch)
            batch["raw_date"] = batch.date.astype("string")
            batch["date"] = _dates(batch.date)
            batch["code"] = batch.code.astype("string").str.strip().str.upper()
            for col in numeric:
                batch[col] = pd.to_numeric(batch[col], errors="coerce").astype(float)
            # Calendar filtering precedes value/duplicate checks. Weekend
            # observations must never become a later as-of update.
            invalid_date = batch.date.isna()
            nontrading = ~invalid_date & ~batch.date.isin(calendar)
            invalid_code = ~batch.code.str.fullmatch(_CODE_PATTERN, na=False)
            for reason, mask in (("invalid_date", invalid_date), ("non_trading_date", nontrading),
                                 ("outside_a_share_code_domain", ~invalid_date & ~nontrading & invalid_code)):
                stats[reason] += int(mask.sum())
                if origin == "market_feature":
                    _quarantine(batch.loc[mask], origin, reason, numeric[0], quarantine)
            batch = batch.loc[~invalid_date & ~nontrading & ~invalid_code].drop(columns="raw_date")
            stats["accepted_key_rows"] += len(batch)
            if not batch.empty:
                first_date = batch.date.min() if first_date is None else min(first_date, batch.date.min())
                last_date = batch.date.max() if last_date is None else max(last_date, batch.date.max())
            for year, group in batch.groupby(batch.date.dt.year, sort=False):
                year = int(year)
                table = pa.Table.from_pandas(group, preserve_index=False)
                if year not in writers:
                    folder.mkdir(parents=True, exist_ok=True)
                    writers[year] = pq.ParquetWriter(folder / f"{year}.parquet", table.schema,
                                                      compression="zstd")
                writers[year].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()
    stats["years"] = sorted(writers)
    return dict(stats) | {"accepted_date_range": [None if item is None else str(item.date())
                                                  for item in (first_date, last_date)]}


def _unique(frame: pd.DataFrame, values: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Conflicting updates survive as null events instead of being skipped."""
    duplicate = frame.duplicated(_KEYS, keep=False)
    duplicates = frame.loc[duplicate]
    if duplicates.empty:
        return frame, duplicates, {"duplicate_rows_removed": 0, "conflicting_keys": 0}
    counts = duplicates.groupby(_KEYS, dropna=False)[values].nunique(dropna=False)
    conflicts = counts.index[counts.gt(1).any(axis=1)]
    result = frame.drop_duplicates(_KEYS, keep="first").copy()
    bad = pd.MultiIndex.from_frame(result[_KEYS]).isin(conflicts)
    result.loc[bad, values] = np.nan
    bad_rows = duplicates.loc[pd.MultiIndex.from_frame(duplicates[_KEYS]).isin(conflicts)].copy()
    result["_duplicate_conflict"] = bad
    return result, bad_rows, {"duplicate_rows_removed": len(frame) - len(result),
                              "conflicting_keys": len(conflicts)}


def _shares(path: Path | None) -> tuple[pd.DataFrame | None, dict]:
    if path is None:
        return None, {"available": False}
    names = _column_names(path)
    required = ["date", "code", "float_shares"]
    if not set(required).issubset(names):
        raise ValueError("share_source requires date, code and float_shares")
    event_cols = [name for name in ("reportdate", "divdate", "listdate", "changedate") if name in names]
    frames = list(_batches(path, required + event_cols, 65536))
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=required + event_cols)
    stats = {"available": True, "source_rows": len(frame), "unit": "ten_thousand_shares",
             "availability_rule": "max(date, reportdate, divdate, listdate, changedate) among supplied event columns"}
    frame["code"] = frame.code.astype("string").str.strip().str.upper()
    valid_code = frame.code.str.fullmatch(_CODE_PATTERN, na=False)
    parsed = pd.DataFrame({col: _dates(frame[col]) for col in ["date"] + event_cols})
    # Missing supplied timestamps cannot establish when an event was known.
    valid_dates = parsed.notna().all(axis=1)
    frame["date"] = parsed.max(axis=1)
    stats["invalid_keys_or_event_dates"] = int((~valid_code | ~valid_dates).sum())
    frame = frame.loc[valid_code & valid_dates, required].copy()
    frame["float_shares"] = pd.to_numeric(frame.float_shares, errors="coerce").astype(float)
    frame, _, duplicates = _unique(frame, ["float_shares"])
    stats.update(duplicates)
    stats["invalid_share_events"] = int((~_positive(frame.float_shares)).sum())
    frame.loc[~_positive(frame.float_shares), "float_shares"] = np.nan
    return frame[_KEYS + ["float_shares"]].sort_values("date"), stats


def _read_year(folder: Path, year: int, values: list[str]) -> tuple[pd.DataFrame, dict]:
    path = folder / f"{year}.parquet"
    if not path.is_file():
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                             "code": pd.Series(dtype="string"),
                             **{col: pd.Series(dtype=float) for col in values}}), {}
    frame = pd.read_parquet(path)
    for col in values:
        if col not in frame:
            frame[col] = np.nan
    frame, _, stats = _unique(frame, values)
    return frame[_KEYS + values], stats


def _crosscheck(frame: pd.DataFrame, shares: pd.DataFrame | None,
                quotes: pd.DataFrame, turnover: pd.DataFrame | None,
                rtol: float, extreme_ratio: float) -> pd.DataFrame:
    result = frame.merge(quotes, on=_KEYS, how="left", validate="one_to_one")
    if shares is not None and not shares.empty:
        result = pd.merge_asof(result.sort_values("date"),
            shares.rename(columns={"date": "share_date"}).sort_values("share_date"),
            left_on="date", right_on="share_date", by="code", direction="backward")
    else:
        result["share_date"], result["float_shares"] = pd.NaT, np.nan
    if turnover is not None:
        result = result.merge(turnover.rename(columns={"turnover": "proxy_turnover"}),
                              on=_KEYS, how="left", validate="one_to_one")
    else:
        result["proxy_turnover"] = result.turnover
    close_good, shares_good = _positive(result.close), _positive(result.float_shares)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result["share_implied_mcap"] = result.close * result.float_shares * 10000.
        result["proxy_mcap"] = result.close * result.volume * 100. / (result.proxy_turnover / 100.)
    result["share_implied_mcap"] = result.share_implied_mcap.where(
        close_good & shares_good & _positive(result.share_implied_mcap))
    result["proxy_mcap"] = result.proxy_mcap.where(
        close_good & _positive(result.volume) & _positive(result.proxy_turnover) & _positive(result.proxy_mcap))
    cap_good = _positive(result.float_mcap)
    result["share_relative_error"] = (result.float_mcap / result.share_implied_mcap - 1).where(cap_good)
    result["proxy_relative_error"] = (result.float_mcap / result.proxy_mcap - 1).where(cap_good)
    share_known, proxy_known = result.share_relative_error.notna(), result.proxy_relative_error.notna()
    disagreement = result.share_relative_error.abs().gt(rtol) | result.proxy_relative_error.abs().gt(rtol)
    result["market_cap_crosscheck_status"] = "not_compared"
    result.loc[share_known | proxy_known, "market_cap_crosscheck_status"] = "consistent_with_available_checks"
    result.loc[disagreement, "market_cap_crosscheck_status"] = "diagnostic_disagreement"
    # Only quarantine enormous, corroborated scale conflicts. A stale share
    # event or rounded turnover alone cannot invalidate the source observation.
    checks_agree = (result.share_implied_mcap / result.proxy_mcap - 1).abs().le(rtol)
    ratio = result.float_mcap / result.share_implied_mcap
    extreme = cap_good & checks_agree & (ratio.ge(extreme_ratio) | ratio.le(1 / extreme_ratio))
    result.loc[extreme, "float_mcap"] = np.nan
    result.loc[extreme, "market_cap_status"] = "extreme_magnitude_conflict"
    result.loc[extreme, "market_cap_crosscheck_status"] = "corroborated_extreme_disagreement"
    return result


def prepare_aiq_market_cap(source: str | Path, calendar: str | Path,
                           output_dir: str | Path, *, share_source: str | Path | None = None,
                           quote_source: str | Path | None = None,
                           proxy_turnover_source: str | Path | None = None,
                           batch_size: int = 65536, comparison_rtol: float = 0.02,
                           extreme_ratio: float = 100.) -> dict:
    """Write an auditable daily cap feed, quarantine and manifest; return manifest.

    ``quote_source`` must provide a reliable **raw** close (and optionally
    volume in lots). ``proxy_turnover_source`` uses percent units and matches
    strictly on date/code; without it, market_feature.turnover is used.
    No price or share extrapolation is performed. All legitimate market-cap
    date/code keys, including invalid-value updates, remain in the main file.
    The caller must as-of match the source date strictly before entry day.
    """
    if batch_size < 1 or not 0 < comparison_rtol < 1 or extreme_ratio <= 1:
        raise ValueError("Invalid batch_size/comparison_rtol/extreme_ratio")
    paths = {"market_feature": Path(source), "calendar": Path(calendar)}
    for name, path in (("shares", share_source), ("quotes", quote_source),
                       ("proxy_turnover", proxy_turnover_source)):
        if path is not None:
            paths[name] = Path(path)
    output_dir = Path(output_dir)
    targets = {name: output_dir / filename for name, filename in
               (("data", "float_mcap.parquet"), ("quarantine", "quarantine.parquet"),
                ("manifest", "manifest.json"))}
    if any(target.resolve() == path.resolve() for target in targets.values() for path in paths.values()):
        raise ValueError("Output cannot overwrite an input source")
    calendar_frames = list(_batches(paths["calendar"], ["date"], batch_size))
    calendar_dates = _dates(pd.concat(calendar_frames, ignore_index=True).date)
    if calendar_dates.isna().any() or calendar_dates.empty:
        raise ValueError("Calendar must contain valid daily dates")
    trading_days = set(calendar_dates)
    shares, share_stats = _shares(paths.get("shares"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stats, statuses, checks = Counter(), Counter(), Counter()
    comparison_totals = Counter()
    annual = []
    with tempfile.TemporaryDirectory(prefix=".aiq_mcap_", dir=output_dir) as temporary:
        stage = Path(temporary)
        quarantine = _Writer(stage / "quarantine.parquet", _QUARANTINE_SCHEMA)
        output = _Writer(stage / "float_mcap.parquet", _OUTPUT_SCHEMA)
        try:
            cap_stats = _partition(paths["market_feature"], _KEYS + ["circulated_market_value"],
                ["turnover"], stage / "cap", trading_days, batch_size, quarantine, "market_feature")
            input_stats = {"market_feature": cap_stats, "shares": share_stats}
            if "quotes" in paths:
                input_stats["quotes"] = _partition(paths["quotes"], _KEYS + ["close"], ["volume"],
                    stage / "quotes", trading_days, batch_size, quarantine, "quotes")
            if "proxy_turnover" in paths:
                input_stats["proxy_turnover"] = _partition(paths["proxy_turnover"], _KEYS + ["turnover"], [],
                    stage / "turnover", trading_days, batch_size, quarantine, "proxy_turnover")
            for year in cap_stats["years"]:
                frame = pd.read_parquet(stage / "cap" / f"{year}.parquet")
                if "turnover" not in frame:
                    frame["turnover"] = np.nan
                # Conflicting turnover alone does not discredit equal cap
                # values: resolve each field's duplicates independently.
                cap, conflicts, duplicate_stats = _unique(frame[_KEYS + ["circulated_market_value"]],
                                                          ["circulated_market_value"])
                turn, _, _ = _unique(frame[_KEYS + ["turnover"]], ["turnover"])
                cap = cap.merge(turn[_KEYS + ["turnover"]], on=_KEYS, validate="one_to_one")
                stats.update(duplicate_stats)
                _quarantine(conflicts, "market_feature", "conflicting_duplicate", "circulated_market_value", quarantine)
                cap["raw_mcap_wan"] = cap.circulated_market_value
                with np.errstate(over="ignore", invalid="ignore"):
                    cap["float_mcap"] = cap.raw_mcap_wan * 10000.
                cap["market_cap_source"] = "aiq_market_feature_circulated_market_value"
                cap["market_cap_status"] = "valid"
                invalid = ~_positive(cap.float_mcap)
                cap.loc[invalid, "market_cap_status"] = "nonfinite_or_nonpositive"
                if "_duplicate_conflict" in cap:
                    cap.loc[cap._duplicate_conflict.fillna(False), "market_cap_status"] = "conflicting_duplicate"
                cap.loc[invalid, "float_mcap"] = np.nan
                quotes, quote_duplicates = _read_year(stage / "quotes", year, ["close", "volume"])
                turnover = None
                turnover_duplicates = {}
                if "proxy_turnover" in paths:
                    turnover, turnover_duplicates = _read_year(stage / "turnover", year, ["turnover"])
                result = _crosscheck(cap, shares, quotes, turnover, comparison_rtol, extreme_ratio)
                result["market_cap_invalid_reason"] = result.market_cap_status.where(
                    result.market_cap_status.ne("valid"), "")
                rejected = result.market_cap_status.ne("valid")
                _quarantine(result.loc[rejected], "market_feature", result.loc[rejected, "market_cap_status"],
                            "raw_mcap_wan", quarantine)
                disagreement = result.market_cap_crosscheck_status.eq("diagnostic_disagreement")
                _quarantine(result.loc[disagreement], "market_feature", "diagnostic_disagreement_not_rejected",
                            "raw_mcap_wan", quarantine)
                result = result.sort_values(_KEYS).reset_index(drop=True)
                output.append(result)
                stats["output_rows"] += len(result)
                statuses.update(result.market_cap_status.value_counts().to_dict())
                checks.update(result.market_cap_crosscheck_status.value_counts().to_dict())
                year_stats = {"year": year, "rows": len(result),
                              "quote_duplicates": quote_duplicates, "turnover_duplicates": turnover_duplicates}
                for name in ("share", "proxy"):
                    error = result[f"{name}_relative_error"].dropna()
                    comparison_totals[f"{name}_compared"] += len(error)
                    comparison_totals[f"{name}_within_tolerance"] += int(error.abs().le(comparison_rtol).sum())
                    year_stats[f"{name}_comparison"] = {"rows": len(error),
                        "median_relative_error": None if error.empty else float(error.median()),
                        "within_tolerance": int(error.abs().le(comparison_rtol).sum())}
                annual.append(year_stats)
        finally:
            output.close()
            quarantine.close()
        manifest = {
            "adapter": "aiq_market_cap_v1", "sources": {name: fingerprint(path) for name, path in paths.items()},
            "outputs": {name: str(path.resolve()) for name, path in targets.items()},
            "units": {"circulated_market_value": "ten_thousand_yuan", "float_mcap": "yuan",
                      "float_shares": "ten_thousand_shares", "volume": "lots_of_100_shares", "turnover": "percent"},
            "unit_evidence": {"share_code": "/data/AIQ/Data/wind_data_api.py:get_share/get_share2 (comments: 万股)",
                "market_field": "get_market_feature_: Wind S_DQ_MV mapped to circulated_market_value; 10000 multiplier corroborated against share events and raw close",
                "limitations": "Local deliveries/formulas are corroborating checks, not independently authenticated vendors."},
            "rules": {"date": "market_feature.date is observation date; never backfill using provider updateTime",
                "pit": "as-of latest record strictly before entry day; retain null events; never skip invalid updates",
                "calendar": "filter source to supplied trading dates before value/duplicate checks",
                "share_asof": share_stats.get("availability_rule"),
                "comparison_rtol": comparison_rtol, "extreme_ratio": extreme_ratio,
                "disagreement": "ordinary mismatches retained and disclosed; extreme scale conflicts rejected only when both formulas agree",
                "prices": "read-only; raw close required; adjusted prices never substituted"},
            "input_stats": input_stats, "counts": dict(stats), "statuses": dict(statuses),
            "crosschecks": dict(checks), "comparison_totals": dict(comparison_totals), "annual": annual,
            "calendar_range": [str(calendar_dates.min().date()), str(calendar_dates.max().date())],
        }
        # All computation completed before publishing; sources stay untouched.
        for name in ("data", "quarantine"):
            (stage / targets[name].name).replace(targets[name])
        write_json(targets["manifest"], manifest)
    return manifest
