"""Point-in-time holdings descriptions for the unified research report.

This module reads frozen holdings and controls. It never changes the backtest
sample, and never uses future-return evaluation weights for holdings profiles.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .config import RISK_FIELDS
from .data import asof_snapshot, columns, dates, fingerprint, read_frame


UNKNOWN_INDUSTRY = "未知行业"


def _source(config: dict, name: str) -> Path | None:
    value = config.get("sources", {}).get(name)
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(config.get("data_root", ".")) / path
    return path


def _industry_names(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    """Use exported names first; do not alias legacy class numbers to AIQ IDs."""
    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for field in ("industry_name", "custom_class_name", "industry"):
        if field not in frame:
            continue
        values = frame[field].astype("string").str.strip().replace("", pd.NA)
        if field == "industry":
            values = values.map(lambda v: mapping.get(str(v), str(v)) if pd.notna(v) else pd.NA)
        result = result.fillna(values)
    return result.fillna(UNKNOWN_INDUSTRY)


def _read_mapping(root: Path) -> dict[str, str]:
    path = root / "industry_mapping.csv"
    if not path.is_file():
        return {}
    mapping = pd.read_csv(path, dtype="string")
    result = {}
    for code_col, name_col in (("industry_code", "industry_name"), ("code", "name")):
        if {code_col, name_col}.issubset(mapping):
            for code, name in mapping[[code_col, name_col]].dropna().itertuples(index=False, name=None):
                result[str(code)] = str(name)
    return result


def _last_risk_per_period(path: Path, cutoffs: pd.Series, fields: list[str], *,
                          aiq_fill: bool = False, calendar: pd.Series | None = None) -> pd.DataFrame:
    """Condense daily snapshots before the asof join, preserving explicit nulls.

    A record belongs to the first observation cutoff at or after its date. The
    last record for each stock in that bucket is enough to reproduce an asof
    join to every cutoff. Subsequent periods can carry that record, including
    explicit missing values, but never borrow a future observation.
    """
    boundaries = np.sort(pd.to_datetime(cutoffs).astype("datetime64[ns]").unique())
    selected = ["date", "code", *fields]
    if not len(boundaries):
        return pd.DataFrame(columns=selected)
    pieces = []
    prior_max = None
    chronologically_ordered_batches = True

    def consume(frame: pd.DataFrame) -> None:
        nonlocal prior_max, chronologically_ordered_batches
        frame["date"] = dates(frame.date)
        if calendar is not None:
            frame = frame.loc[frame.date.isin(calendar)].copy()
        frame["_bucket"] = np.searchsorted(boundaries, frame.date.to_numpy(dtype="datetime64[ns]"), side="left")
        frame = frame.loc[frame._bucket.lt(len(boundaries))]
        if frame.empty:
            return
        if prior_max is not None and frame.date.min() < prior_max:
            chronologically_ordered_batches = False
        prior_max = max(prior_max, frame.date.max()) if prior_max is not None else frame.date.max()
        frame = frame.sort_values("date", kind="stable")
        if aiq_fill:
            # AIQ unlisted risk fields use ffill: only NaN is skipped. Values
            # such as zero, negatives, or infinity remain genuine donors.
            for field in fields:
                frame[field + "__donor"] = frame.date.where(frame[field].notna())
            frame = frame.groupby(["_bucket", "code"], sort=False, as_index=False).last()
        else:
            frame = frame.drop_duplicates(["_bucket", "code"], keep="last")
        pieces.append(frame)

    if path.suffix == ".feather":
        with pa.memory_map(str(path), "r") as stream:
            reader = ipc.open_file(stream)
            for i in range(reader.num_record_batches):
                consume(reader.get_batch(i).select(selected).to_pandas())
    elif path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(columns=selected, batch_size=65536):
            consume(batch.to_pandas())
    else:
        consume(read_frame(path, selected))
    if not pieces:
        return pd.DataFrame(columns=selected)
    compact = pd.concat(pieces, ignore_index=True)
    compact = compact.sort_values("date", kind="stable")
    if aiq_fill:
        if chronologically_ordered_batches:
            compact = compact.groupby(["_bucket", "code"], sort=False, as_index=False).last()
        else:
            # A source need not be sorted physically. Choose by each donor's
            # date, not by a different field's later record in the same batch.
            keys = ["_bucket", "code"]
            latest = compact.drop_duplicates(keys, keep="last").set_index(keys)[["date"]]
            for field in fields:
                donor = field + "__donor"
                best = (compact.sort_values(donor, na_position="first", kind="stable")
                        .drop_duplicates(keys, keep="last").set_index(keys)[[field, donor]])
                latest = latest.join(best, how="left", validate="one_to_one")
            compact = latest.reset_index()
        compact = compact.sort_values(["code", "date"], kind="stable")
        payload = fields + [field + "__donor" for field in fields]
        compact[payload] = compact.groupby("code", sort=False)[payload].ffill()
        return compact.drop(columns="_bucket")
    return compact.drop_duplicates(["_bucket", "code"], keep="last").drop(columns="_bucket")


def _risk_controls(base: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    available = [name for name in RISK_FIELDS if name in base]
    path = _source(config, "risk")
    meta = {"fields": list(RISK_FIELDS), "normalization": "eligible universe mean and population standard deviation at each entry",
            "time_policy": "last risk record on or before the frozen cutoff; explicit null records are preserved"}
    if len(available) < len(RISK_FIELDS) and path is not None and path.is_file():
        source_columns = columns(path)
        missing = [field for field in RISK_FIELDS if field not in base and field in source_columns]
        if missing:
            aiq_fill = config.get("missing_values", {}).get("method") == "aiq_matrix"
            calendar_path = _source(config, "calendar")
            calendar = (read_frame(calendar_path, ["date"]).date
                        if aiq_fill and calendar_path is not None and calendar_path.is_file() else None)
            compact = _last_risk_per_period(path, base.cutoff, missing, aiq_fill=aiq_fill, calendar=calendar)
            joined = asof_snapshot(base[["entry", "code", "cutoff"]], compact, "profile_risk")
            if (joined.profile_risk_date > joined.cutoff).any():
                raise ValueError("Report risk snapshot exceeds the signal cutoff")
            base = base.merge(joined[["entry", "code", "profile_risk_date", *missing]], on=["entry", "code"],
                              how="left", validate="one_to_one")
            available += missing
            meta["source"] = fingerprint(path)
            meta["matched_rows"] = int(joined.profile_risk_date.notna().sum())
            meta["missing_snapshot_rows"] = int(joined.profile_risk_date.isna().sum())
            if aiq_fill:
                meta["time_policy"] = "AIQ ffill per risk field on the trading calendar; true donor dates on or before cutoff"
                meta["missing_values_method"] = "aiq_matrix"
                meta["calendar"] = fingerprint(calendar_path) if calendar is not None else None
                meta["value_sources"] = {}
                boundary = pd.to_datetime(joined.cutoff).dt.normalize()
                if calendar is not None:
                    grid = np.sort(calendar.to_numpy(dtype="datetime64[ns]"))
                    pos = np.searchsorted(grid, joined.cutoff.to_numpy(dtype="datetime64[ns]"), side="right") - 1
                    boundary = pd.Series(pd.NaT, index=joined.index, dtype="datetime64[ns]")
                    good = pos >= 0
                    boundary.loc[good] = grid[pos[good]]
                for field in missing:
                    donor = joined[field + "__donor"]
                    if (donor > joined.cutoff).any():
                        raise ValueError("Report AIQ risk donor exceeds cutoff")
                    meta["value_sources"][field] = {
                        "observed_or_carried_rows": int(joined[field].notna().sum()),
                        "filled_rows": int((joined[field].notna() & donor.lt(boundary)).sum()),
                        "earliest_donor": donor.min().strftime("%Y-%m-%d") if donor.notna().any() else None,
                        "latest_donor": donor.max().strftime("%Y-%m-%d") if donor.notna().any() else None,
                    }
            del compact, joined
    for field in RISK_FIELDS:
        if field not in base:
            base[field] = np.nan
        values = pd.to_numeric(base[field], errors="coerce").replace([np.inf, -np.inf], np.nan)
        grouping = values.groupby(base.entry, sort=False)
        average = grouping.transform("mean")
        std = grouping.transform("std", ddof=0)
        base[field] = ((values - average) / std.where(std.gt(1e-12))).astype("float32")
    meta["available_fields"] = available
    if not available:
        meta["status"] = "risk_source_unavailable"
    else:
        meta["status"] = "ready"
    return base, meta


def _cap_categories(base: pd.DataFrame) -> pd.Series:
    """Equal-count rank bins of actual positive cap; tied values stay together."""
    values = pd.to_numeric(base.float_mcap, errors="coerce")
    values = values.where(np.isfinite(values) & values.gt(0))
    percentile = values.groupby(base.entry, sort=False).rank(method="average", pct=True)
    return (np.ceil(percentile * 5) - 1).clip(0, 4)


def _profile(frame: pd.DataFrame, weight_column: str = "target_weight") -> dict:
    """Aggregate frozen target weights; all missingness remains explicit."""
    w = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0).to_numpy(dtype=float)
    if (w < -1e-12).any() or w.sum() > 1 + 1e-7:
        raise ValueError("Report profile requires a long-only target portfolio of total weight at most one")
    weights = pd.Series(w, index=frame.index)
    industry = weights.groupby(frame["_industry"], dropna=False).sum()
    industry = {str(k): round(float(v), 8) for k, v in industry.items() if v > 0}
    style = frame[RISK_FIELDS].to_numpy(dtype=float)
    valid = np.isfinite(style)
    covered = (valid * w[:, None]).sum(axis=0)
    numerator = (np.where(valid, style, 0) * w[:, None]).sum(axis=0)
    styles = {name: round(float(numerator[i] / covered[i]), 6) if covered[i] > 0 else None
              for i, name in enumerate(RISK_FIELDS)}
    style_coverage = {name: round(float(covered[i]), 8) for i, name in enumerate(RISK_FIELDS)}
    cap = pd.to_numeric(frame.float_mcap, errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(cap) & (cap > 0) & (w > 0)
    cap_weight = float(w[good].sum())
    median = None
    if cap_weight > 0:
        order = np.argsort(cap[good], kind="stable")
        cumulative = np.cumsum(w[good][order])
        median = round(float(cap[good][order][np.searchsorted(cumulative, cap_weight / 2, side="left")]), 2)
    bins = [round(float(w[frame._cap_bin.eq(i).to_numpy()].sum()), 8) for i in range(5)]
    return {"industries": industry, "styles": styles, "style_coverage": style_coverage,
            "cap_bins": bins, "cap_coverage": round(cap_weight, 8), "cap_median": median,
            "cash": round(max(0., 1 - float(w.sum())), 8),
            "industry_coverage": round(float(weights.loc[frame._industry.ne(UNKNOWN_INDUSTRY)].sum()), 8),
            "stocks": int((w > 0).sum())}


def _empty_profile() -> dict:
    frame = pd.DataFrame(columns=["target_weight", "_industry", "float_mcap", "_cap_bin", *RISK_FIELDS])
    return _profile(frame)


def _eligible_controls(root: Path, mapping: dict[str, str]) -> pd.DataFrame:
    universe_path, controls_path = root / "universe.parquet", root / "controls.parquet"
    if not universe_path.is_file():
        return pd.DataFrame()
    univ_columns = columns(universe_path)
    needed = [field for field in ("entry", "code", "cutoff", "eligible") if field in univ_columns]
    universe = pd.read_parquet(universe_path, columns=needed)
    if "eligible" in universe:
        universe = universe.loc[universe.eligible.fillna(False)].drop(columns="eligible")
    universe["entry"] = pd.to_datetime(universe.entry)
    if "cutoff" not in universe:
        universe["cutoff"] = universe.entry - pd.Timedelta(nanoseconds=1)
    if universe.duplicated(["entry", "code"]).any():
        raise ValueError("Duplicate eligible universe keys for report profiles")
    if controls_path.is_file():
        ctrl_columns = columns(controls_path)
        needed = [field for field in ("entry", "code", "industry_name", "custom_class_name", "industry", "float_mcap", *RISK_FIELDS)
                  if field in ctrl_columns]
        controls = pd.read_parquet(controls_path, columns=needed)
        universe = universe.merge(controls, on=["entry", "code"], how="left", validate="one_to_one")
    universe["_industry"] = _industry_names(universe, mapping)
    if "float_mcap" not in universe:
        universe["float_mcap"] = np.nan
    universe["_cap_bin"] = _cap_categories(universe)
    return universe


def _signature(root: Path, config: dict, factors: list[str]) -> dict:
    paths = [root / filename for filename in ("universe.parquet", "controls.parquet", "industry_mapping.csv")]
    paths += [root / "factors" / factor / filename for factor in factors
              for filename in ("holdings.parquet", "portfolio_periodic.csv", "portfolio_weekly.csv")]
    paths += [path for key in ("risk", "calendar") if (path := _source(config, key)) is not None]
    paths += [Path(__file__)]
    return {"config_digest": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "files": {str(path): fingerprint(path) if path.is_file() else None for path in paths}}


def build_profiles(root: Path, config: dict, factor_names: list[str], *, use_cache: bool = True) -> dict:
    """Describe Q1/Q10/ALL and the eligible universe without rerunning returns."""
    root = Path(root).resolve()
    signature = _signature(root, config, factor_names)
    cache = root / "report_profiles.json"
    if use_cache and cache.is_file():
        try:
            saved = json.loads(cache.read_text())
            if saved.get("_meta", {}).get("inputs") == signature:
                return saved
        except (ValueError, OSError):
            pass
    result = {name: {} for name in factor_names}
    mapping = _read_mapping(root)
    base = _eligible_controls(root, mapping)
    result["_meta"] = {
        "weight": "frozen target_weight; exit rows removed; future-return evaluation weights are never used",
        "style_weighting": "weighted mean among observed exposures, accompanied by observed target-weight coverage for every field",
        "cap_bins": "five cross-sectional average-rank bins of positive actual float market cap in the eligible universe",
        "cap_unit": "CNY",
        "industry_mapping": str(root / "industry_mapping.csv"),
        "reference": "ALL is the available sample for each factor/variant; it is not necessarily the cross-variant intersection",
        "sample_policy": config.get("neutralization", {}).get("sample_policy", "common"),
        "status": "ready",
        "inputs": signature,
    }
    result["_universe"] = {}
    if base.empty:
        result["_meta"]["status"] = "eligible_universe_unavailable"
        return result
    base, risk_meta = _risk_controls(base, config)
    result["_meta"]["risk"] = risk_meta
    # Universe counts are independent of factor/Barra availability.
    base["target_weight"] = 1 / base.groupby("entry", sort=False).code.transform("size")
    for date, frame in base.groupby("entry", sort=True):
        result["_universe"][date.strftime("%Y-%m-%d")] = _profile(frame)
    base = base[["entry", "code", "_industry", "float_mcap", "_cap_bin", *RISK_FIELDS]]
    for factor in factor_names:
        path = root / "factors" / factor / "holdings.parquet"
        if not path.is_file():
            result["_meta"].setdefault("missing_holdings", []).append(factor)
            continue
        schema = columns(path)
        if not {"entry", "code", "variant", "portfolio", "target_weight"}.issubset(schema):
            result["_meta"].setdefault("invalid_holdings", []).append(factor)
            continue
        variants = config.get("neutralization", {}).get("variants")
        if not variants:
            variants = pd.read_parquet(path, columns=["variant"]).variant.dropna().unique()
        # Missing/empty target groups still receive an explicit cash profile.
        weekly_path = root / "factors" / factor / "portfolio_periodic.csv"
        if not weekly_path.is_file():
            weekly_path = root / "factors" / factor / "portfolio_weekly.csv"
        schedule = pd.read_csv(weekly_path) if weekly_path.is_file() else pd.DataFrame()
        for variant in variants:
            result[factor][str(variant)] = {}
            filters = [("variant", "==", variant), ("portfolio", "in", ["Q1", "Q10", "ALL"]), ("target_weight", ">", 0)]
            if "is_exit" in schema:
                filters.append(("is_exit", "==", False))
            holdings = pd.read_parquet(path, columns=["entry", "code", "portfolio", "target_weight"], filters=filters)
            if holdings.duplicated(["entry", "portfolio", "code"]).any():
                raise ValueError("Duplicate frozen target holdings in report profile")
            merged = holdings.merge(base, on=["entry", "code"], how="left", validate="many_to_one")
            merged["_industry"] = merged._industry.fillna(UNKNOWN_INDUSTRY)
            for (date, portfolio), frame in merged.groupby(["entry", "portfolio"], sort=True):
                key = pd.Timestamp(date).strftime("%Y-%m-%d")
                result[factor][str(variant)].setdefault(key, {})[str(portfolio)] = _profile(frame)
            if not schedule.empty:
                subset = schedule.loc[schedule.variant.eq(variant) & schedule.portfolio.isin(["Q1", "Q10", "ALL"])]
                for row in subset.to_dict("records"):
                    day = pd.Timestamp(row["entry"]).strftime("%Y-%m-%d")
                    portfolio = row["portfolio"]
                    # Do not silently invent holdings for a nonempty or missing-signal portfolio.
                    if portfolio not in result[factor][str(variant)].setdefault(day, {}):
                        value = _empty_profile()
                        if row.get("n_stocks") == 0 and row.get("cash_weight") == 1:
                            value["status"] = "cash"
                        else:
                            value.update(status="unavailable_holdings", cash=None, cap_coverage=None,
                                         industry_coverage=None, cap_bins=[None] * 5,
                                         style_coverage={field: None for field in RISK_FIELDS})
                        result[factor][str(variant)][day][portfolio] = value
            del holdings, merged
    if use_cache:
        temporary = cache.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        temporary.replace(cache)
    return result
