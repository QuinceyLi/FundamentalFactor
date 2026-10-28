from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as parquet


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(v):
        if v is pd.NaT or v is pd.NA:
            return None
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.integer, np.floating, np.bool_)):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, (pd.Timestamp, Path)):
            return str(v)
        return v

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        h.update(stream.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            stream.seek(max(1024 * 1024, stat.st_size - 1024 * 1024))
            h.update(stream.read())
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sampled_sha256": h.hexdigest(), "hash_method": "first/last 1 MiB + size + mtime_ns (not full hash)"}


def columns(path: Path) -> list[str]:
    if path.suffix == ".feather":
        import pyarrow as pa
        import pyarrow.ipc as ipc
        with pa.memory_map(str(path), "r") as src:
            return ipc.open_file(src).schema.names
    if path.suffix == ".parquet":
        return parquet.read_schema(path).names
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    raise ValueError(f"Unsupported input format: {path}")


def dates(values: pd.Series, nullable: bool = False) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        result = pd.to_datetime(values)
    elif pd.api.types.is_integer_dtype(values.dtype):
        # Mongo's repeated YYYYMMDD integers can be parsed directly with a
        # fixed format; avoid constructing millions of temporary strings.
        if not values.dropna().between(10000101, 99991231).all():
            raise ValueError("Integer dates must be eight-digit YYYYMMDD")
        result = pd.to_datetime(values, format="%Y%m%d", errors="raise")
    else:
        # Both int YYYYMMDD and ISO strings are supported; integers must never be ns epochs.
        text = values.astype("string").str.replace(r"\.0$", "", regex=True)
        compact = text.str.fullmatch(r"\d{8}", na=False)
        result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        result.loc[compact] = pd.to_datetime(text[compact], format="%Y%m%d", errors="raise")
        result.loc[~compact] = pd.to_datetime(text[~compact], errors="raise")
    if getattr(result.dt, "tz", None) is not None:
        raise ValueError("Date snapshots must be timezone-naive calendar dates")
    if (result.dropna() != result.dropna().dt.normalize()).any():
        raise ValueError("Daily date columns must not contain intraday timestamps")
    if not nullable and result.isna().any():
        raise ValueError("Null date key")
    return result


def read_frame(path: Path, selected: list[str] | None = None, *, nullable_booleans: bool = False) -> pd.DataFrame:
    if path.suffix == ".feather":
        frame = feather.read_table(path, columns=selected).to_pandas()
    elif path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=selected)
    elif path.suffix == ".csv":
        frame = pd.read_csv(path, usecols=selected, dtype={"code": "string"})
    else:
        raise ValueError(f"Unsupported input format: {path}")
    for name in ("date", "listing_date", "delisting_date", "available_date", "notice_date", "record_date"):
        if name in frame:
            frame[name] = dates(frame[name], nullable=name in ("delisting_date", "record_date"))
    if "code" in frame:
        frame["code"] = frame.code.astype("string").str.strip().str.upper()
        if not frame.code.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)", na=False).all():
            raise ValueError(f"Noncanonical A-share code in {path}")
    for name in ("is_a_share", "is_st", "suspended", "limit_up", "limit_down"):
        if name in frame:
            if (not nullable_booleans and frame[name].isna().any()) or not frame[name].dropna().isin([True, False, 0, 1]).all():
                raise ValueError(f"{name} must contain explicit non-null booleans or 0/1")
            frame[name] = frame[name].astype("boolean" if nullable_booleans else bool)
    return frame


def require_unique(frame: pd.DataFrame, keys: list[str]) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Null/duplicate keys: {keys}")


def asof_snapshot(left: pd.DataFrame, right: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Last *record*, not last nonmissing value; explicit null updates are preserved."""
    left, right = left.copy(), right.copy()
    left["code"], right["code"] = left.code.astype("string"), right.code.astype("string")
    # Pandas/Arrow can preserve seconds or milliseconds in otherwise identical
    # timestamp columns. A common precision prevents incompatible merge keys
    # while retaining the nanosecond immediately-before-entry signal cutoff.
    left["cutoff"] = pd.to_datetime(left.cutoff).astype("datetime64[ns]")
    right["date"] = pd.to_datetime(right.date).astype("datetime64[ns]")
    right = right.rename(columns={"date": f"{prefix}_date"})
    return pd.merge_asof(left.sort_values("cutoff"), right.sort_values(f"{prefix}_date"),
                         left_on="cutoff", right_on=f"{prefix}_date", by="code",
                         direction="backward", allow_exact_matches=True)
