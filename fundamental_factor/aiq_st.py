"""Auditable ST state candidates from AIQ's cached event and name intervals.

The untyped AShareST extract includes events other than ST.  Its ANN_DT is
not necessarily an original ST announcement.  Historical names corroborate
states only inside their own coverage; an old open name is never extended
through a newer event extract.  This is a historical vendor/name proxy, not
a certified point-in-time announcement archive.
"""
from __future__ import annotations

from pathlib import Path
import unicodedata

import pandas as pd

from .data import fingerprint, require_unique, write_json
from .st_history import st_from_names


def _dates(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce").dt.normalize()
    text = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.to_datetime(text.where(compact), format="%Y%m%d", errors="coerce")
    other = pd.to_datetime(text.where(~compact), errors="coerce")
    return parsed.fillna(other).dt.normalize()


def _day(value) -> pd.Timestamp:
    parsed = _dates(pd.Series([value])).iloc[0]
    if pd.isna(parsed):
        raise ValueError("Coverage date is invalid")
    return parsed


def _text(value) -> str:
    return "" if pd.isna(value) else str(value)


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"code": "string"})
    raise ValueError("AIQ ST sources must be CSV or XLSX; executable caches are unsupported")


def _normalize(raw: pd.DataFrame, kind: str, quarantine: list[dict]):
    required = {"code", "execute_date", "cancel_date", "name" if kind == "names" else "announce_date"}
    if not required.issubset(raw):
        raise ValueError(f"AIQ {kind} source is missing columns: {sorted(required - set(raw))}")
    frame = raw.copy().reset_index(drop=True)
    frame["_row"] = frame.index + 2  # CSV/Excel row including its header.
    frame["code"] = frame.code.astype("string").str.strip().str.upper()
    frame["_start"] = _dates(frame.execute_date)
    frame["_cancel"] = _dates(frame.cancel_date)
    frame["_announce"] = (_dates(frame.announce_date) if kind == "events" else pd.NaT)
    frame["_end"] = frame._cancel + pd.Timedelta(days=1 if kind == "names" else 0)
    uncertain_codes = set()

    def record(row, reason, severity="excluded"):
        quarantine.append({"source_kind": kind, "source_row": int(row._row),
                           "code": _text(row.code), "execute_date": _text(row.execute_date),
                           "cancel_date": _text(row.cancel_date),
                           "announce_date": _text(row.get("announce_date")),
                           "name": _text(row.get("name")), "reason": reason, "severity": severity})

    valid_code = frame.code.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)", na=False)
    missing_end = frame.cancel_date.isna() | frame.cancel_date.astype("string").str.strip().isin(["", "-", "--"])
    invalid_end = ~missing_end & frame._cancel.isna()
    invalid_dates = frame._start.isna() | invalid_end | frame._end.le(frame._start)
    bad = ~valid_code | invalid_dates
    for _, row in frame.loc[bad].iterrows():
        record(row, "invalid_code_or_interval")
        if valid_code.loc[row.name]:
            uncertain_codes.add(row.code)
    frame = frame.loc[~bad].copy()
    duplicates = frame.duplicated(sorted(required), keep="first")
    for _, row in frame.loc[duplicates].iterrows():
        record(row, "exact_duplicate_collapsed", "deduplicated")
    frame = frame.loc[~duplicates].copy()
    if kind == "events":
        for _, row in frame.loc[frame._announce.isna()].iterrows():
            record(row, "missing_or_invalid_announcement_not_used_for_availability", "warning")
        for _, row in frame.loc[frame._announce.gt(frame._start)].iterrows():
            record(row, "announcement_after_effective_date_semantics_unverified", "warning")
    else:
        frame["_name_st"] = st_from_names(frame["name"])
        clean = frame["name"].map(lambda x: unicodedata.normalize("NFKC", str(x))).str.upper()
        clean = clean.str.replace(r"\s+", "", regex=True)
        frame["_special_name"] = clean.str.contains("退", regex=False) | clean.str.match(r"^(?:G)?PT")
    return frame, uncertain_codes


def _coverage(frame: pd.DataFrame, supplied, label: str) -> pd.Timestamp:
    # Latest effective event is a disclosed coverage proxy, not a download or
    # first-publication timestamp. An override may shorten, never extend it.
    evidence_end = frame._start.max()
    if pd.isna(evidence_end):
        raise ValueError(f"AIQ {label} source has no valid dated rows")
    result = evidence_end if supplied is None else _day(supplied)
    if result > evidence_end:
        raise ValueError(f"{label} coverage cannot extend beyond the last dated source evidence")
    return result


def build_aiq_st_events(events_raw: pd.DataFrame, names_raw: pd.DataFrame, *,
                        st_valid_through=None, names_valid_through=None):
    """Return state transitions, quarantine, per-code coverage and metadata.

    Name BEGINDATE/ENDDATE are inclusive; ST ENTRY_DT is inclusive and
    REMOVE_DT exclusive. Dates stay on their recorded natural day. No name
    is inferred before its own start, and unknown tombstones stop as-of
    joins at interval gaps and each source's expiry.
    """
    quarantine: list[dict] = []
    events, uncertain_events = _normalize(events_raw, "events", quarantine)
    names, uncertain_names = _normalize(names_raw, "names", quarantine)
    event_stop = _coverage(events, st_valid_through, "events")
    name_stop = _coverage(names, names_valid_through, "names")
    metadata = {"event_coverage_through": event_stop, "name_coverage_through": name_stop,
                "event_source_rows": len(events_raw), "name_source_rows": len(names_raw),
                "event_rows_after_exact_deduplication": len(events),
                "name_rows_after_exact_deduplication": len(names),
                "event_duplicate_start_rows": int(events_raw.duplicated(["code", "execute_date"]).sum()),
                "announcement_after_execute_rows": int((_dates(events_raw.announce_date) >
                                                         _dates(events_raw.execute_date)).sum())}
    for frame, stop in ((events, event_stop), (names, name_stop)):
        frame["_end"] = frame._end.fillna(stop + pd.Timedelta(days=1)).clip(upper=stop + pd.Timedelta(days=1))
    events = events.loc[events._start.le(event_stop)]
    names = names.loc[names._start.le(name_stop)]
    event_groups = dict(tuple(events.groupby("code", sort=False)))
    name_groups = dict(tuple(names.groupby("code", sort=False)))
    codes = sorted(set(event_groups) | set(name_groups))
    records, coverage = [], []
    for code in codes:
        ev = event_groups.get(code, events.iloc[:0])
        nm = name_groups.get(code, names.iloc[:0])
        boundaries = sorted(set(ev._start) | set(ev._end) | set(nm._start) | set(nm._end))
        code_records = []
        for date in boundaries:
            ae = ev.loc[ev._start.le(date) & ev._end.gt(date)]
            an = nm.loc[nm._start.le(date) & nm._end.gt(date)]
            value, priority = pd.NA, 0
            if code in uncertain_events | uncertain_names:
                reason = "invalid_source_interval_for_code"
            elif len(an) > 1 and an["name"].nunique(dropna=False) > 1:
                reason = "conflicting_name_intervals"
            elif an.empty:
                reason = ("name_coverage_expired" if date > name_stop else
                          "untyped_event_without_name" if not ae.empty else "name_interval_gap")
            elif an._special_name.any():
                reason = "retirement_or_other_special_name_not_classified_as_st"
            elif an._name_st.isna().any():
                reason = "missing_name_state"
            elif an._name_st.all():
                value = True
                reason = "st_event_and_name_agree" if not ae.empty else "st_name_only_proxy"
                priority = 3 if not ae.empty else 2
            elif not ae.empty:
                reason = "untyped_event_conflicts_with_non_st_name"
            else:
                value, priority, reason = False, 2, "non_st_name_proxy_no_active_event"
            if date > max(event_stop, name_stop):
                reason = "source_coverage_expired"
            known = not pd.isna(value)
            event_ids = ";".join(map(str, sorted(ae._row.tolist())))
            name_ids = ";".join(map(str, sorted(an._row.tolist())))
            source = "AIQ:AShareST+ASharePreviousName" if event_ids and name_ids else (
                "AIQ:ASharePreviousName" if name_ids else "AIQ:AShareST" if event_ids else "AIQ:coverage")
            row = {"date": date, "code": code, "is_st": value, "st_known": known,
                   "status": reason, "source": source, "st_source": source,
                   "st_status_reason": reason, "st_evidence_priority": priority,
                   "is_proxy": True, "st_source_rows": event_ids, "name_source_rows": name_ids,
                   "name": "|".join(sorted(an["name"].dropna().astype(str).unique())),
                   "st_execute_dates": ";".join(sorted(ae._start.dt.strftime("%Y-%m-%d").unique())),
                   "st_cancel_dates": ";".join(sorted(ae._cancel.dropna().dt.strftime("%Y-%m-%d").unique())),
                   "st_announce_dates": ";".join(sorted(ae._announce.dropna().dt.strftime("%Y-%m-%d").unique())),
                   "st_source_valid_through": event_stop, "name_source_valid_through": name_stop,
                   "availability_basis": "historical_effective_name_interval_proxy_not_first_publication"}
            code_records.append(row)
            if reason in {"conflicting_name_intervals", "untyped_event_conflicts_with_non_st_name",
                          "untyped_event_without_name", "retirement_or_other_special_name_not_classified_as_st"}:
                quarantine.append({"source_kind": "joined_state", "code": code,
                                   "execute_date": date.strftime("%Y-%m-%d"), "source_row": event_ids,
                                   "name": row["name"], "reason": reason, "severity": "unknown_state"})
        records.extend(code_records)
        known_rows = [r for r in code_records if r["st_known"]]
        coverage.append({"code": code, "first_source_date": min(boundaries),
                         "last_state_boundary": max(boundaries),
                         "first_known_state_date": min((r["date"] for r in known_rows), default=pd.NaT),
                         "has_known_state": bool(known_rows), "event_intervals": len(ev),
                         "name_intervals": len(nm), "name_coverage_through": name_stop,
                         "event_coverage_through": event_stop})
    result = pd.DataFrame(records).sort_values(["date", "code"]).reset_index(drop=True)
    result["is_st"] = result.is_st.astype("boolean")
    result["st_known"] = result.st_known.astype(bool)
    result["st_name"] = result["name"]
    require_unique(result, ["date", "code"])
    rejected = pd.DataFrame(quarantine).reindex(columns=["source_kind", "source_row", "code", "execute_date",
                                                        "cancel_date", "announce_date", "name", "reason", "severity"])
    return result, rejected, pd.DataFrame(coverage), metadata


def prepare_aiq_st(st_source: str | Path, names_source: str | Path, output: str | Path, *,
                   st_valid_through=None, names_valid_through=None,
                   calendar_source: str | Path | None = None) -> dict:
    """Write independent Parquet states plus manifest, coverage and quarantine.

    An optional one-column ``date`` CSV/Parquet calendar is checked and
    fingerprinted, but never used to move an effective event backwards.
    Unknown AIQ rows should only be fallback candidates, never erase an
    independently known official state during a multi-source merge.
    """
    st_source, names_source, output = map(lambda p: Path(p).resolve(), (st_source, names_source, output))
    if output.suffix != ".parquet" or output in (st_source, names_source):
        raise ValueError("AIQ ST export requires a separate .parquet output")
    if output.is_relative_to(st_source.parent) or output.is_relative_to(names_source.parent):
        raise ValueError("AIQ source directories are read-only; use a separate artifact directory")
    frame, quarantine, coverage, summary = build_aiq_st_events(
        _load(st_source), _load(names_source), st_valid_through=st_valid_through,
        names_valid_through=names_valid_through)
    summary["sources"] = {"st": fingerprint(st_source), "names": fingerprint(names_source)}
    if calendar_source is not None:
        calendar_source = Path(calendar_source).resolve()
        cal = (pd.read_parquet(calendar_source, columns=["date"]) if calendar_source.suffix == ".parquet"
               else pd.read_csv(calendar_source, usecols=["date"]))
        calendar = _dates(cal.date)
        if calendar.isna().any() or calendar.duplicated().any():
            raise ValueError("Calendar contains invalid or duplicate dates")
        summary["calendar"] = {"source": fingerprint(calendar_source), "start": calendar.min(),
                               "end": calendar.max(), "dates": len(calendar),
                               "non_session_event_boundaries_retained": int((~frame.date.isin(calendar)).sum())}
    output.parent.mkdir(parents=True, exist_ok=True)
    coverage_path = output.with_suffix(".coverage.parquet")
    quarantine_path = output.with_suffix(".quarantine.csv")
    manifest_path = output.with_suffix(".manifest.json")
    frame.to_parquet(output, index=False)
    coverage.to_parquet(coverage_path, index=False)
    quarantine.to_csv(quarantine_path, index=False)
    summary.update({"output": str(output), "output_fingerprint": fingerprint(output),
                    "coverage_path": str(coverage_path), "quarantine_path": str(quarantine_path),
                    "manifest_path": str(manifest_path), "rows": len(frame), "codes": frame.code.nunique(),
                    "known_state_rows": int(frame.st_known.sum()),
                    "known_st_rows": int(frame.is_st.fillna(False).sum()),
                    "known_codes": int(coverage.has_known_state.sum()),
                    "statuses": frame.st_status_reason.value_counts().to_dict(),
                    "quarantine_reasons": quarantine.reason.value_counts().to_dict(),
                    "interval_policy": "name start/end inclusive; ST start inclusive/cancel exclusive; natural dates unchanged",
                    "coverage_policy": "each source expires the day after its latest effective event, or an earlier explicit cutoff",
                    "limitations": [
                        "Historical vendor/name proxy, not certified first-publication PIT availability.",
                        "AShareST export filters S_TYPE_ST != R but omits the actual type; unmatched events stay unknown.",
                        "ANN_DT is preserved for audit, never used to shift the effective start or treated as the first announcement.",
                        "Source coverage ceilings are evidence-date proxies, not independently verified refresh timestamps.",
                        "Old open name intervals expire at the name source ceiling, independently of the newer ST source.",
                        "No whole-universe non-ST default; missing codes, interval gaps and conflicts remain unknown.",
                        "get_preST predicts reporting-risk announcements, not an actual ST classification, and is not imported.",
                        "Names containing retirement/PT markers cannot establish an ST state.",
                        "Only exact duplicate rows are collapsed; overlapping untyped events retain source row provenance."]})
    write_json(manifest_path, summary)
    return summary
