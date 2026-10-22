"""Evidence-based, constant-segment repair of an adjusted-price discontinuity.

The raw price and adjustment-factor files may share an upstream vendor. They
are compared as separate local deliveries, not advertised as independent data
vendors. Public observations provide a separate check of the identified event.
Nothing here infers missing corporate actions or edits an original source.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .data import require_unique, write_json


@dataclass(frozen=True)
class SegmentPatch:
    code: str
    event_date: int
    previous_date: int
    multiplier: float
    overlap_before: int
    overlap_after: int
    legacy_factor_before: float
    legacy_factor_after: float
    reference_factor_before: float
    reference_factor_after: float


def compact_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if pd.api.types.is_datetime64_any_dtype(frame.date):
        if getattr(frame.date.dt, "tz", None) is not None or not frame.date.eq(frame.date.dt.normalize()).all():
            raise ValueError("Price dates must be naive daily dates")
        frame["date"] = frame.date.dt.strftime("%Y%m%d").astype("int64")
    if frame.date.isna().any() or not frame.date.astype(str).str.fullmatch(r"\d{8}").all():
        raise ValueError("Expected YYYYMMDD price dates")
    pd.to_datetime(frame.date.astype(str), format="%Y%m%d", errors="raise")
    frame["date"] = frame.date.astype("int64")
    require_unique(frame, ["date", "code"])
    return frame


def factor_chain_audit(frame: pd.DataFrame, tolerance: float = 1e-6) -> tuple[pd.DataFrame, dict]:
    """Identify deviations, not errors: restructurings can add valid factors.

    The usual reference-price adjustment is A[t]/A[t-1] = close[t-1]/last[t].
    A deviation is insufficient evidence to delete an adjustment event.
    """
    frame = compact_dates(frame).sort_values(["code", "date"]).reset_index(drop=True)
    for column in ("close", "last", "acc_adj_factor"):
        if not (np.isfinite(frame[column]) & frame[column].gt(0)).all():
            raise ValueError(f"Invalid {column}: factor chain cannot be verified")
    grouped = frame.groupby("code", observed=True)
    frame["prev_close"] = grouped.close.shift()
    frame["prev_factor"] = grouped.acc_adj_factor.shift()
    frame["prev_date"] = grouped.date.shift()
    frame["factor_step"] = frame.acc_adj_factor / frame.prev_factor
    frame["implied_factor_step"] = frame.prev_close / frame["last"]
    frame["residual"] = frame.factor_step / frame.implied_factor_step - 1
    anomalous = frame.prev_factor.notna() & frame.residual.abs().gt(tolerance)
    columns = ["date", "code", "last", "close", "acc_adj_factor", "prev_date", "prev_close",
               "prev_factor", "factor_step", "implied_factor_step", "residual"]
    return frame.loc[anomalous, columns].copy(), {
        "rows": len(frame), "codes": frame.code.nunique(), "tolerance": tolerance,
        "deviations": int(anomalous.sum()),
        "negative_deviations": int((anomalous & frame.residual.lt(0)).sum()),
        "max_abs_residual": float(frame.residual.abs().max()),
        "interpretation": "Deviations require review; positive restructuring adjustments are not automatically errors.",
    }


def derive_segment_patch(legacy: pd.DataFrame, reference: pd.DataFrame, code: str,
                         event_date: int, tolerance: float = 1e-6) -> SegmentPatch:
    """Require matching raw quotes and constant relative scales on both sides.

    Anchor the post-event legacy scale. Rescale *all* earlier adjusted prices
    of this security by one constant. This preserves every return wholly
    inside either segment and avoids splicing unrelated factor levels.
    """
    old = compact_dates(legacy[legacy.code.eq(code)])
    new = compact_dates(reference[reference.code.eq(code)])
    if old.empty or new.empty:
        raise ValueError("Both deliveries must contain the selected security")
    fields = ["date", "code", "open", "close", "last", "acc_adj_factor"]
    overlap = old[fields].merge(new[fields], on=["date", "code"], suffixes=("_old", "_new"))
    overlap = overlap.sort_values("date")
    for field in ("open", "close", "last"):
        if not np.isclose(overlap[field + "_old"], overlap[field + "_new"], rtol=1e-8, atol=1e-8).all():
            raise ValueError(f"Raw {field} disagreement: automatic segment repair forbidden")
    for suffix in ("_old", "_new"):
        value = overlap["acc_adj_factor" + suffix]
        if not (np.isfinite(value) & value.gt(0)).all():
            raise ValueError("Adjustment factors must be finite and positive")
    before = overlap[overlap.date.lt(event_date)]
    after = overlap[overlap.date.ge(event_date)]
    if len(before) < 2 or len(after) < 2 or after.date.iloc[0] != event_date:
        raise ValueError("Need at least two matching observations on both sides, including event date")
    previous, current = before.iloc[-1], after.iloc[0]
    if int(old[old.date.lt(event_date)].date.max()) != int(previous.date):
        raise ValueError("Reference feed does not include the preceding legacy observation")
    if int(new[new.date.lt(event_date)].date.max()) != int(previous.date):
        raise ValueError("Legacy feed does not include the preceding reference observation")
    ratios_before = before.acc_adj_factor_old / before.acc_adj_factor_new
    ratios_after = after.acc_adj_factor_old / after.acc_adj_factor_new
    ratio_before, ratio_after = float(ratios_before.iloc[-1]), float(ratios_after.iloc[0])
    if not np.allclose(ratios_before, ratio_before, rtol=tolerance, atol=0):
        raise ValueError("Pre-event relative factor scale is not constant")
    if not np.allclose(ratios_after, ratio_after, rtol=tolerance, atol=0):
        raise ValueError("Post-event relative factor scale is not constant")
    if np.isclose(ratio_before, ratio_after, rtol=tolerance, atol=0):
        raise ValueError("No independently confirmed scale discontinuity")
    expected = previous.close_new / current.last_new
    observed = current.acc_adj_factor_new / previous.acc_adj_factor_new
    if not np.isclose(observed, expected, rtol=tolerance, atol=0):
        raise ValueError("Reference event factor does not match the raw reference-price adjustment")
    return SegmentPatch(code, int(event_date), int(previous.date), ratio_after / ratio_before,
        len(before), len(after), float(previous.acc_adj_factor_old), float(current.acc_adj_factor_old),
        float(previous.acc_adj_factor_new), float(current.acc_adj_factor_new))


def repair_adjusted_table(table: pa.Table, patch: SegmentPatch) -> tuple[pa.Table, pd.DataFrame]:
    """Preserve Arrow schema, index metadata, keys and every unaffected value."""
    if not np.isfinite(patch.multiplier) or patch.multiplier <= 0:
        raise ValueError("Invalid repair multiplier")
    price_fields = [name for name in table.column_names if name in
                    ("acc_open", "acc_close", "acc_high", "acc_low", "acc_vwap", "acc_last")]
    if not {"acc_open", "acc_close"}.issubset(price_fields):
        raise ValueError("Adjusted open and close are required")
    mask = pc.and_(pc.equal(table["code"], patch.code), pc.less(table["date"], patch.event_date))
    affected = table.filter(mask)
    if affected.num_rows == 0:
        raise ValueError("Repair matches no source rows")
    ledger = affected.select(["date", "code"] + price_fields).to_pandas().reset_index(drop=True)
    ledger = ledger.rename(columns={name: name + "_original" for name in price_fields})
    ledger["multiplier"] = patch.multiplier
    for name in price_fields:
        if not pa.types.is_floating(table[name].type):
            raise ValueError(f"Expected floating adjusted price column: {name}")
        column = pc.if_else(mask, pc.multiply(table[name], patch.multiplier), table[name])
        table = table.set_column(table.column_names.index(name), table.schema.field(name), column)
        ledger[name + "_repaired"] = ledger[name + "_original"] * patch.multiplier
    return table, ledger


def full_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sina_dividend_rows(path: Path) -> list[dict]:
    """Read the named dividend table, retaining the source's per-10-share unit."""
    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth, self.cells, self.cell, self.rows = 0, [], None, []

        def handle_starttag(self, tag, attrs):
            if tag == "table" and (self.depth or dict(attrs).get("id") == "sharebonus_1"):
                self.depth += 1
            if self.depth and tag == "tr":
                self.cells = []
            if self.depth and tag == "td":
                self.cell = []

        def handle_data(self, text):
            if self.cell is not None:
                self.cell.append(text)

        def handle_endtag(self, tag):
            if tag == "td" and self.cell is not None:
                self.cells.append("".join(self.cell).strip())
                self.cell = None
            if tag == "tr" and self.depth and len(self.cells) >= 8:
                self.rows.append(self.cells)
            if tag == "table" and self.depth:
                self.depth -= 1

    parser = TableParser()
    parser.feed(path.read_bytes().decode("gb18030"))
    records = []
    for row in parser.rows:
        if row[4] == "实施" and row[5] != "--":
            date = int(pd.Timestamp(row[5]).strftime("%Y%m%d"))
            records.append({"notice_date": row[0], "ex_date": date,
                "bonus_shares_per_10": float(row[1]), "transfer_shares_per_10": float(row[2]),
                "cash_per_10_shares": float(row[3])})
    if not records:
        raise ValueError("No implemented dividends parsed from the archived Sina table")
    return records


def public_event_evidence(raw: Path, reference: pd.DataFrame, patch: SegmentPatch) -> dict:
    """Validate archived responses, raw OHLC agreement and event-day coverage.

    Eastmoney's adjusted-price formula differs from this local factor formula;
    its adjusted prices are only corroborating evidence, never replacement data.
    """
    records = json.loads((raw / "fetch_manifest.json").read_text(encoding="utf-8"))
    sources = {record["file"]: record for record in records}
    required = ("sina_000001_dividends.html", "eastmoney_000001_raw.json", "eastmoney_000001_hfq.json")
    for name in required:
        record = sources.get(name, {})
        if record.get("status") != 200 or full_sha256(raw / name) != record.get("sha256"):
            raise ValueError(f"Public evidence response/hash not verified: {name}")

    def bars(name: str) -> pd.DataFrame:
        response = json.loads((raw / name).read_text(encoding="utf-8"))
        data = response.get("data") or {}
        if response.get("rc") != 0 or data.get("code") != patch.code.split(".")[0]:
            raise ValueError("Public kline response status/security mismatch")
        rows = [line.split(",") for line in data.get("klines", [])]
        frame = pd.DataFrame([[row[0], row[1], row[2]] for row in rows], columns=["date", "open", "close"])
        if frame.empty:
            raise ValueError("Public kline evidence is empty")
        frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y%m%d").astype("int64")
        frame[["open", "close"]] = frame[["open", "close"]].astype(float)
        require_unique(frame, ["date"])
        return frame

    unadjusted = bars("eastmoney_000001_raw.json")
    adjusted = bars("eastmoney_000001_hfq.json")
    local = compact_dates(reference[reference.code.eq(patch.code)])
    compared = unadjusted.merge(local[["date", "open", "close"]], on="date", suffixes=("_public", "_local"))
    if len(compared) != len(unadjusted):
        raise ValueError("Public evidence includes observations absent from the local reference")
    for field in ("open", "close"):
        if not np.isclose(compared[field + "_public"], compared[field + "_local"], atol=1e-8, rtol=0).all():
            raise ValueError(f"Independent public raw {field} does not match the local reference")
    adjusted = adjusted.set_index("date")
    unadjusted = unadjusted.set_index("date")
    for frame in (adjusted, unadjusted):
        if not {patch.previous_date, patch.event_date}.issubset(frame.index):
            raise ValueError("Public evidence does not cover both sides of the repair event")
    dividends = sina_dividend_rows(raw / "sina_000001_dividends.html")
    if any(row["ex_date"] == patch.event_date for row in dividends):
        raise ValueError("A public dividend event coincides with the disputed date; review repair manually")
    adjacent_dividend = min((row for row in dividends if row["ex_date"] > patch.event_date),
                            key=lambda row: row["ex_date"], default=None)
    if adjacent_dividend is None:
        raise ValueError("Public dividend table does not cover the event period")
    return {
        "sources": [sources[name] for name in required],
        "independent_raw_rows_compared": len(compared), "raw_open_close_disagreements": 0,
        "event_previous_close": float(unadjusted.loc[patch.previous_date, "close"]),
        "event_close": float(unadjusted.loc[patch.event_date, "close"]),
        "public_raw_close_return": float(unadjusted.loc[patch.event_date, "close"] /
                                         unadjusted.loc[patch.previous_date, "close"] - 1),
        "public_adjusted_close_return": float(adjusted.loc[patch.event_date, "close"] /
                                              adjusted.loc[patch.previous_date, "close"] - 1),
        "sina_adjacent_dividend": adjacent_dividend,
        "sina_dividend_on_repair_date": False,
        "scope": "Public raw observations corroborate the event. Public adjusted levels are not spliced into local data.",
    }


def prepare_repaired_prices(output: Path, mongo_path: Path, legacy_path: Path,
                            reference_path: Path) -> dict:
    """Audit three full local files and stage the single publicly verified repair."""
    output = output.resolve()
    target = output / "validated"
    target.mkdir(parents=True, exist_ok=True)
    result_path = target / "quotefac_repaired.feather"
    inputs = {"mongo": mongo_path.resolve(), "legacy": legacy_path.resolve(), "reference": reference_path.resolve()}
    if result_path.resolve() in inputs.values():
        raise ValueError("Original source files must never be overwritten")
    fingerprints = {name: {"path": str(path), "bytes": path.stat().st_size,
                           "sha256": full_sha256(path)} for name, path in inputs.items()}
    fields = ["date", "code", "open", "close", "last", "volume", "acc_adj_factor"]
    legacy = pd.read_feather(legacy_path, columns=fields + ["acc_open", "acc_close"])
    # The selected research universe is Shanghai/Shenzhen, starting in 2013.
    legacy = legacy[legacy.date.ge(20130101) & legacy.code.str.fullmatch(r"\d{6}\.(SH|SZ)")]
    reference = compact_dates(pd.read_feather(reference_path, columns=fields))
    legacy = compact_dates(legacy)
    chain_stats = {}
    for label, frame in (("legacy", legacy), ("reference", reference)):
        deviations, chain_stats[label] = factor_chain_audit(frame)
        deviations.to_csv(output / (label + "_factor_anomalies.csv"), index=False)
    patch = derive_segment_patch(legacy, reference, "000001.SZ", 20250528)
    independent = public_event_evidence(output / "raw", reference, patch)

    overlap = legacy[fields].merge(reference[fields], on=["date", "code"], suffixes=("_old", "_new"))
    comparison = {"local_delivery_overlap_rows": len(overlap), "raw_disagreements": {}}
    for field in ("open", "close", "last", "volume"):
        comparison["raw_disagreements"][field] = int((~np.isclose(overlap[field + "_old"],
            overlap[field + "_new"], rtol=1e-8, atol=1e-8)).sum())
    overlap["factor_ratio"] = overlap.acc_adj_factor_old / overlap.acc_adj_factor_new
    range_by_code = overlap.groupby("code").factor_ratio.agg(["min", "max"])
    comparison["nonconstant_factor_ratio_codes"] = range_by_code[
        ~np.isclose(range_by_code["min"], range_by_code["max"], rtol=1e-6, atol=0)].reset_index().to_dict("records")
    del overlap

    table = feather.read_table(mongo_path)
    mongo = table.select(["date", "code", "acc_open", "acc_close"]).to_pandas().reset_index(drop=True)
    require_unique(mongo, ["date", "code"])
    comparison["mongo_rows"] = len(mongo)
    paired = mongo.merge(legacy[["date", "code", "acc_open", "acc_close"]],
                         on=["date", "code"], suffixes=("_mongo", "_legacy"), how="left", indicator=True)
    matched = paired._merge.eq("both")
    comparison["mongo_legacy_overlap_rows"] = int(matched.sum())
    comparison["mongo_legacy_adjusted_disagreements"] = {field: int((matched &
        ~np.isclose(paired[field + "_mongo"], paired[field + "_legacy"], rtol=1e-8, atol=1e-8)).sum())
        for field in ("acc_open", "acc_close")}
    missing = paired.loc[~matched, ["date", "code"]].merge(reference[["date", "code"]],
        on=["date", "code"], how="left", indicator=True)
    missing = missing.loc[missing._merge.ne("both"), ["date", "code"]]
    missing.to_csv(output / "mongo_no_raw_source.csv", index=False)
    comparison["mongo_keys_without_either_raw_delivery"] = {"rows": len(missing),
        "codes": missing.code.nunique(), "handling": "Retained, not fabricated or zero-filled; raw trading status is unknown."}
    del paired
    recent_pair = mongo.merge(reference[["date", "code", "open", "close", "acc_adj_factor"]], on=["date", "code"])
    comparison["mongo_reference_overlap_rows"] = len(recent_pair)
    expected_problem = recent_pair.code.eq(patch.code) & recent_pair.date.lt(patch.event_date)
    comparison["mongo_reference_other_disagreements_above_rounding"] = {}
    for field in ("open", "close"):
        error = (recent_pair["acc_" + field] - recent_pair[field] * recent_pair.acc_adj_factor).abs()
        comparison["mongo_reference_other_disagreements_above_rounding"][field] = int((~expected_problem & error.gt(0.000051)).sum())
    if any(comparison["mongo_legacy_adjusted_disagreements"].values()):
        raise ValueError("Mongo adjusted prices differ from the legacy delivery; repair assumptions need review")
    if any(comparison["mongo_reference_other_disagreements_above_rounding"].values()):
        raise ValueError("Additional Mongo/reference discrepancies require review before publishing")

    repaired, ledger = repair_adjusted_table(table, patch)
    feather.write_feather(repaired, result_path)
    ledger.to_feather(target / "quotefac_repair_patches.feather")
    roundtrip = feather.read_table(result_path)
    if not roundtrip.equals(repaired, check_metadata=True):
        raise ValueError("Repaired Feather failed exact round-trip validation")
    event_before = mongo[mongo.code.eq(patch.code) & mongo.date.isin([patch.previous_date, patch.event_date])].set_index("date")
    prior_return = float(event_before.loc[patch.event_date, "acc_close"] / event_before.loc[patch.previous_date, "acc_close"] - 1)
    fixed_return = float((prior_return + 1) / patch.multiplier - 1)
    manifest = {"schema_version": 1, "repair_type": "constant_segment_rescaling",
        "source_files": fingerprints, "local_comparison": comparison, "factor_chain_scan": chain_stats,
        "patch": asdict(patch), "adjusted_rows": len(ledger),
        "price_fields": [name[:-9] for name in ledger if name.endswith("_original")],
        "event_close_return_before": prior_return, "event_close_return_after": fixed_return,
        "public_evidence": independent,
        "return_definition": "Vendor reference-price adjusted-price return; not asserted to be exact cash-dividend total return.",
        "method": "Anchor the legacy post-event scale; multiply all pre-event acc_* price levels for one security by a constant. Other securities, post-event prices and non-price fields are unchanged.",
        "unmodified_special_events": "Nine positive factor/reference-price deviations remain as supplied. They may reflect restructuring or shareholder consideration and are not automatically treated as corrupt prices.",
        "source_truth_boundary": "Local files may share an upstream vendor. Public raw quotes independently confirm only the audited event window. No missing status, corporate action, or delisting settlement has been invented.",
        "artifacts": [{"file": name, "sha256": full_sha256(target / name), "bytes": (target / name).stat().st_size}
                      for name in ("quotefac_repaired.feather", "quotefac_repair_patches.feather")]}
    # Check input identity again so concurrent source changes cannot produce a
    # repair attributed to a different version of the input data.
    if any(full_sha256(path) != fingerprints[name]["sha256"] for name, path in inputs.items()):
        raise ValueError("A source changed during repair; staged files must not be published")
    write_json(target / "quotefac_repair_manifest.json", manifest)
    return manifest
