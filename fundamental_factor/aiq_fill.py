"""AIQ-compatible value filling without importing or executing the AIQ project.

``apply_fill`` mirrors Tools/datahub.py's pure matrix operation. The long-table
and snapshot interfaces additionally retain originals, fill markers and actual
value provenance. Calendar=None means event snapshots; supplying a calendar
means the AIQ daily grid, including its exclusion of off-calendar records.
Lifecycle/coverage masks belong to the caller, after filling.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import dates


FILL_ZERO, FILL_KEEP, FILL_FFILL, FILL_LAST = "zero", "keep", "ffill", "last"
FILL_RULES = {
    **dict.fromkeys(("volume", "amount", "return", "vwap"), FILL_ZERO),
    **dict.fromkeys(("turnover", "free_turnover", "open", "high", "low", "zdtmark", "fy1", "CAGR"), FILL_KEEP),
    **{f"{prefix}_{minute}toe": FILL_LAST
       for prefix in ("vwap", "twap") for minute in (30, 60, 90, 120, 150, 180, 210)},
}
_POLICIES = {FILL_ZERO, FILL_KEEP, FILL_FFILL, FILL_LAST}
_META = ("original", "original_missing", "filled", "value_source_date", "fill_anchor_date", "fill_method")


def field_policy(field: str, fill_rules: Mapping[str, str] | None = None) -> str:
    """Unlisted columns, including adjusted prices and market cap, use ffill."""
    policy = (fill_rules or {}).get(field, FILL_RULES.get(field, FILL_FFILL))
    if policy not in _POLICIES:
        raise ValueError(f"Unsupported fill policy for {field}: {policy}")
    return policy


def apply_fill(values: pd.DataFrame, policy: str,
               fill_source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Exact AIQ _apply_fill semantics; no finite/positive-value filtering.

    AIQ's numeric matrix construction converts boolean columns to floating
    point before this operation. The zero policy expects such numeric values.
    The last reference is already a matrix, normally FP.last (itself ffilled).
    """
    if policy == FILL_KEEP:
        return values
    if policy == FILL_ZERO:
        return values.mask(values.ffill().notna(), 1) * values.fillna(0)
    if policy == FILL_LAST:
        if fill_source is None:
            return values.ffill()
        return values.fillna(fill_source.reindex(index=values.index, columns=values.columns))
    return values.ffill()


def _calendar(calendar: pd.Series | pd.Index | pd.DataFrame | Sequence | None):
    if calendar is None:
        return None
    values = calendar["date"] if isinstance(calendar, pd.DataFrame) else pd.Series(calendar)
    result = pd.DatetimeIndex(dates(values)).sort_values()
    if result.has_duplicates:
        raise ValueError("Calendar contains duplicate dates")
    return result


def _prepare(frame: pd.DataFrame, fields: list[str], calendar=None, codes=None) -> pd.DataFrame:
    missing = {"date", "code", *fields} - set(frame)
    if missing:
        raise ValueError(f"Missing fill input columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = dates(result.date).astype("datetime64[ns]")
    result["code"] = result.code.astype("string")
    if result.code.isna().any():
        raise ValueError("Null stock code")
    if calendar is not None:
        result = result[result.date.isin(calendar)]
    if codes is not None:
        result = result[result.code.isin(codes)]
    if result.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate (date, code) fill records")
    return result.sort_values(["date", "code"], kind="stable").reset_index(drop=True)


def _validate_fields(fields: Sequence[str]) -> list[str]:
    fields = list(fields)
    if not fields or len(fields) != len(set(fields)) or {"date", "code"} & set(fields):
        raise ValueError("fields must contain distinct non-key columns")
    return fields


def _backward(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Return source payload in target row order, including missing target dates."""
    payload = [col for col in source if col != "code"]
    good = target["_target_date"].notna()
    matched = pd.merge_asof(target.loc[good].sort_values("_target_date", kind="stable"),
        source.sort_values("date", kind="stable"), left_on="_target_date", right_on="date",
        by="code", direction="backward", allow_exact_matches=True)
    return matched.set_index("_position")[payload].reindex(range(len(target)))


def _last_reference(source: pd.DataFrame, fill_source: pd.DataFrame | None,
                    last_field: str, calendar, codes) -> pd.DataFrame | None:
    if fill_source is not None:
        return _prepare(fill_source, [last_field], calendar, codes)[["date", "code", last_field]]
    if last_field in source:
        return source[["date", "code", last_field]]
    return None


def _resolved(target: pd.DataFrame, original: pd.Series, original_date: pd.Series,
              source: pd.DataFrame, field: str, policy: str,
              reference: pd.DataFrame | None, last_field: str) -> dict[str, pd.Series]:
    """Resolve only requested targets: a full stock×day matrix is unnecessary."""
    observed = original.notna()
    value = original.copy()
    value_date = original_date.where(observed).copy()
    anchor = value_date.copy()
    method = pd.Series("missing", index=original.index, dtype="string")
    method.loc[observed] = "observed"
    if policy == FILL_KEEP:
        pass
    elif policy == FILL_LAST and reference is not None:
        # FP.last is a default-ffill column. Missing latest reference records
        # must therefore not prevent using an earlier non-NA reference value.
        valid = reference.loc[reference[last_field].notna()]
        candidates = _backward(target, valid)
        fill = ~observed & candidates[last_field].notna()
        value.loc[fill] = candidates.loc[fill, last_field]
        value_date.loc[fill] = candidates.loc[fill, "date"]
        anchor.loc[fill] = candidates.loc[fill, "date"]
        method.loc[fill] = "last"
    else:
        valid = source.loc[source[field].notna(), ["date", "code", field]]
        candidates = _backward(target, valid)
        fill = ~observed & candidates[field].notna()
        anchor.loc[fill] = candidates.loc[fill, "date"]
        if policy == FILL_ZERO:
            value.loc[fill] = False if pd.api.types.is_bool_dtype(value.dtype) else 0
            # A generated zero is a rule result, not a source observation.
            value_date.loc[fill] = pd.NaT
            method.loc[fill] = "zero"
        else:
            value.loc[fill] = candidates.loc[fill, field]
            value_date.loc[fill] = candidates.loc[fill, "date"]
            method.loc[fill] = "ffill"  # includes the documented last-source fallback
    return {field: value, f"{field}_original": original,
            f"{field}_original_missing": ~observed,
            f"{field}_filled": ~observed & value.notna(),
            f"{field}_value_source_date": pd.to_datetime(value_date),
            f"{field}_fill_anchor_date": pd.to_datetime(anchor),
            f"{field}_fill_method": method}


def filled_snapshot(base: pd.DataFrame, source: pd.DataFrame, fields: Sequence[str], *,
                    cutoff: str = "cutoff", prefix: str = "source",
                    calendar=None, fill_rules: Mapping[str, str] | None = None,
                    last_field: str = "last", fill_source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach AIQ-filled snapshots, with originals and provenance, to base.

    Without calendar, use the latest known event: its explicit NA may ffill,
    but an unchanged event is not an absent daily observation. With calendar,
    use the last grid date <= cutoff and apply each field's daily missing-cell
    rule. Off-calendar source rows are excluded only in that explicit mode.

    All source metadata outside fields comes from the latest original record,
    unchanged and without value filling. ``prefix_date``/``prefix_record_date``
    identify that record, not a reconstructed observation. Each field records
    its true value_source_date. For generated zeros it is NaT and the last real
    observation appears only in fill_anchor_date. No bfill is performed.
    """
    fields = _validate_fields(fields)
    if {cutoff, "code"} - set(base):
        raise ValueError("Snapshot base requires code and cutoff")
    cal = _calendar(calendar)
    codes = base.code.astype("string")
    if codes.isna().any():
        raise ValueError("Null snapshot stock code")
    known = pd.to_datetime(base[cutoff]).reset_index(drop=True)
    if known.isna().any() or getattr(known.dt, "tz", None) is not None:
        raise ValueError("Snapshot cutoffs must be non-null and timezone-naive")
    known = known.astype("datetime64[ns]")
    src = _prepare(source, fields, cal, codes.unique())
    # Restrict work to requested securities and dates without changing any
    # retained observation. This keeps sparse snapshots useful on large feeds.
    src = src.loc[src.date.le(known.max())]
    policies = {field: field_policy(field, fill_rules) for field in fields}
    reference = (_last_reference(src, fill_source, last_field, cal, codes.unique())
                 if FILL_LAST in policies.values() else None)
    if cal is None:
        target_date = known
        grid_date = pd.Series(pd.NaT, index=known.index, dtype="datetime64[ns]")
    else:
        locations = cal.searchsorted(known, side="right") - 1
        target_date = pd.Series(pd.NaT, index=known.index, dtype="datetime64[ns]")
        usable = locations >= 0
        target_date.loc[usable] = cal.to_numpy()[locations[usable]]
        grid_date = target_date.copy()
    target = pd.DataFrame({"_position": np.arange(len(base)), "code": codes.to_numpy(),
                           "_target_date": target_date})
    target["code"] = target.code.astype("string")
    matched = _backward(target, src)
    payload = [col for col in source if col not in ("date", "code")]
    # As with a source join, latest source metadata is authoritative. Field
    # fill metadata is appended separately, never silently applied to statuses.
    result = base.drop(columns=[col for col in payload if col in base], errors="ignore").copy()
    original_index = result.index
    result = result.reset_index(drop=True)
    for col in payload:
        result[col] = matched[col]
    result[f"{prefix}_date"] = matched.date
    result[f"{prefix}_record_date"] = matched.date
    result[f"{prefix}_grid_date"] = grid_date
    exact = matched.date.eq(target_date) if cal is not None else pd.Series(True, index=matched.index)
    for field in fields:
        original = matched[field].where(exact)
        meta = _resolved(target, original, matched.date, src, field, policies[field], reference, last_field)
        for col, values in meta.items():
            result[col] = values
    result.index = original_index
    return result


def fill_long_records(frame: pd.DataFrame, fields: Sequence[str], calendar=None, *,
                      codes: Sequence[str] | None = None,
                      fill_rules: Mapping[str, str] | None = None,
                      last_field: str = "last", fill_source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fill existing event rows, or an explicit daily calendar×codes grid.

    No calendar means no invented rows. Supplying a calendar explicitly asks
    for the dense grid; use filled_snapshot for a large feed when only a few
    rebalance dates are needed. Result keys are ordered by date/code. The
    source frame is not mutated. Nullable strings/booleans are preserved.
    """
    fields = _validate_fields(fields)
    cal = _calendar(calendar)
    code_list = list(pd.unique(frame.code.astype("string"))) if codes is None else list(codes)
    if len(code_list) != len(set(code_list)):
        raise ValueError("Duplicate grid codes")
    source = _prepare(frame, fields, cal, code_list)
    if cal is None:
        keys = source[["date", "code"]].copy()
    else:
        keys = pd.MultiIndex.from_product([cal, code_list], names=["date", "code"]).to_frame(index=False)
        keys = keys.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    base = keys.rename(columns={"date": "cutoff"})
    result = filled_snapshot(base, source, fields, cutoff="cutoff", prefix="source",
        calendar=cal, fill_rules=fill_rules, last_field=last_field, fill_source=fill_source)
    result = result.rename(columns={"cutoff": "date"})
    return result
