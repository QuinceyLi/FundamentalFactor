"""Point-in-time cross-sectional transformations; no return labels are accepted."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import qr

from .config import DEFAULT_PREPROCESS, RISK_FIELDS, VARIANTS


def standardize(values: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    std = float(centered.std(ddof=0))
    if std <= tolerance * max(1., float(np.abs(values).max())):
        return np.zeros(len(values))
    return centered / std


def winsorize(values: np.ndarray, settings: dict | None = None) -> tuple[np.ndarray, str]:
    settings = DEFAULT_PREPROCESS | (settings or {})
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        low, high = np.quantile(values, settings["fallback_quantiles"])
        method = "quantile_fallback"
    else:
        width = settings["mad_multiplier"] * settings["mad_normalizer"] * mad
        low, high = median - width, median + width
        method = "mad"
    return np.clip(values, low, high), method


def control_matrix(frame: pd.DataFrame, full: bool = False, *,
                   include_float_mcap: bool = True) -> tuple[np.ndarray, list[str]]:
    industry = pd.get_dummies(frame.industry, prefix="industry", drop_first=True, dtype=float)
    parts = [np.ones((len(frame), 1)), industry.to_numpy()]
    names = ["intercept"] + industry.columns.tolist()
    if include_float_mcap:
        parts.append(standardize(np.log(frame.float_mcap.to_numpy()))[:, None])
        names.append("log_float_mcap")
    if full:
        parts.extend(standardize(frame[col].to_numpy())[:, None] for col in RISK_FIELDS)
        names.extend(RISK_FIELDS)
    return np.column_stack(parts), names


def neutralize(values: np.ndarray, controls: np.ndarray, names: list[str],
               tolerance: float = 1e-10) -> tuple[np.ndarray, dict]:
    q, r, pivots = qr(controls, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    rank = int((diagonal > tolerance * max(float(diagonal.max(initial=0.)), 1.)).sum())
    info = {"rank": rank, "columns": len(names),
            "redundant_columns": ",".join(names[i] for i in pivots[rank:])}
    if len(values) <= rank + 1:
        raise ValueError("insufficient_cross_section_degrees_of_freedom")
    basis = q[:, :rank]
    residual = values - basis @ (basis.T @ values)
    return residual, info


def prepare_signals(frame: pd.DataFrame, settings: dict | None = None, *,
                    include_float_mcap: bool = True,
                    variants: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = DEFAULT_PREPROCESS | (settings or {})
    variants = list(variants if variants is not None else
                    (VARIANTS if include_float_mcap else ["raw", "industry", "full"]))
    if (not variants or len(set(variants)) != len(variants) or
            set(variants) - {"raw", "industry", "industry_size", "full"}):
        raise ValueError("variants must be nonempty unique supported transformations")
    needs_industry = any(v != "raw" for v in variants)
    needs_cap = "industry_size" in variants or ("full" in variants and include_float_mcap)
    numeric_columns = ["value"] + (["float_mcap"] if needs_cap else []) + (RISK_FIELDS if "full" in variants else [])
    required = ["entry", "exit", "code"] + (["industry"] if needs_industry else []) + numeric_columns
    if missing := set(required) - set(frame):
        raise ValueError(f"preprocessing missing columns: {sorted(missing)}")
    if frame.duplicated(["entry", "exit", "code"]).any():
        raise ValueError("duplicate preprocessing keys")
    signals, diagnostics = [], []
    for entry, group in frame.groupby("entry", sort=True):
        group = group.sort_values("code").copy()
        numeric = group[numeric_columns].to_numpy(dtype=float)
        if (not np.isfinite(numeric).all() or (needs_industry and group.industry.isna().any())
                or (needs_cap and group.float_mcap.le(0).any())):
            raise ValueError("prepare_signals requires a frozen complete common sample")
        clipped, method = winsorize(group.value.to_numpy(), settings)
        raw = standardize(clipped, settings["rank_tolerance"])
        transformed = {"raw": raw}
        infos = {"raw": {"rank": 0, "columns": 0, "redundant_columns": ""}}
        failed = None
        for variant in (v for v in variants if v != "raw"):
            use_cap = variant == "industry_size" or (variant == "full" and include_float_mcap)
            controls, names = control_matrix(group, variant == "full", include_float_mcap=use_cap)
            try:
                residual, info = neutralize(raw, controls, names, settings["rank_tolerance"])
                transformed[variant] = standardize(residual, settings["rank_tolerance"])
                infos[variant] = info
            except ValueError as exc:
                failed = str(exc)
                break
        if failed:
            diagnostics.append({"entry": entry, "variant": "all", "status": "failed",
                                "reason": failed, "n_stocks": len(group), "winsor_method": method})
            continue
        for variant in variants:
            out = group.copy()
            out["variant"], out["signal"] = variant, transformed[variant]
            out["percentile"] = out.signal.rank(method="average", pct=True)
            out["group"] = np.ceil(10 * out.percentile).clip(1, 10).astype(int)
            signals.append(out)
            constant = np.ptp(transformed[variant]) == 0
            diagnostics.append({"entry": entry, "variant": variant,
                                "status": "constant" if constant else "ok", "reason": "constant_signal" if constant else "",
                                "n_stocks": len(group), "winsor_method": method, **infos[variant]})
    signal_columns = list(frame.columns) + ["variant", "signal", "percentile", "group"]
    diag_columns = ["entry", "variant", "status", "reason", "n_stocks", "winsor_method",
                    "rank", "columns", "redundant_columns"]
    return (pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(columns=signal_columns),
            pd.DataFrame(diagnostics, columns=diag_columns))
