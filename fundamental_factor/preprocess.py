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
                   include_float_mcap: bool = True, market_cap_column: str = "float_mcap",
                   include_industry: bool = True) -> tuple[np.ndarray, list[str]]:
    parts = [np.ones((len(frame), 1))]
    names = ["intercept"]
    if include_industry:
        industry = pd.get_dummies(frame.industry, prefix="industry", drop_first=True, dtype=float)
        parts.append(industry.to_numpy())
        names.extend(industry.columns.tolist())
    if include_float_mcap:
        capital = frame[market_cap_column].to_numpy(dtype=float)
        if not np.isfinite(capital).all() or (capital <= 0).any():
            raise ValueError(f"{market_cap_column} must be finite and positive")
        parts.append(standardize(np.log(capital))[:, None])
        names.append(f"log_{market_cap_column}")
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
                    variants: list[str] | None = None,
                    sample_policy: str = "common",
                    market_cap_column: str = "float_mcap") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform precomputed factors without using realized returns.

    The legacy common policy requires a complete shared sample. Independent
    samples filter only the inputs needed by each transformation, so requesting
    a risk-neutral result cannot change the raw result or suppress its period.
    """
    settings = DEFAULT_PREPROCESS | (settings or {})
    for option in ("winsorize", "standardize"):
        if not isinstance(settings.get(option, True), bool):
            raise ValueError(f"preprocess.{option} must be boolean")
    if sample_policy not in {"common", "independent"}:
        raise ValueError("sample_policy must be common or independent")
    if not isinstance(market_cap_column, str) or not market_cap_column:
        raise ValueError("market_cap_column must be a nonempty column name")
    variants = list(variants if variants is not None else
                    (VARIANTS if include_float_mcap else ["raw", "industry", "full"]))
    if (not variants or len(set(variants)) != len(variants) or
            set(variants) - {"raw", "size", "industry", "industry_size", "full"}):
        raise ValueError("variants must be nonempty unique supported transformations")
    needs_industry = any(v in {"industry", "industry_size", "full"} for v in variants)
    needs_cap = bool({"size", "industry_size"} & set(variants)) or ("full" in variants and include_float_mcap)
    numeric_columns = ["value"] + ([market_cap_column] if needs_cap else []) + (RISK_FIELDS if "full" in variants else [])
    required = ["entry", "exit", "code"] + (["industry"] if needs_industry else []) + numeric_columns
    if sample_policy == "independent":
        required = ["entry", "exit", "code", "value"]
    if missing := set(required) - set(frame):
        raise ValueError(f"preprocessing missing columns: {sorted(missing)}")
    if frame.duplicated(["entry", "exit", "code"]).any():
        raise ValueError("duplicate preprocessing keys")
    signals, diagnostics = [], []

    def transform_base(values):
        if settings.get("winsorize", True):
            values, method = winsorize(values, settings)
        else:
            values, method = np.asarray(values, dtype=float).copy(), "disabled"
        if settings.get("standardize", True):
            values = standardize(values, settings["rank_tolerance"])
        return values, method

    def transform_neutral(values, group, variant):
        use_cap = variant in {"size", "industry_size"} or (variant == "full" and include_float_mcap)
        controls, names = control_matrix(group, variant == "full", include_float_mcap=use_cap,
                                        market_cap_column=market_cap_column,
                                        include_industry=variant != "size")
        residual, info = neutralize(values, controls, names, settings["rank_tolerance"])
        if settings.get("standardize", True):
            residual = standardize(residual, settings["rank_tolerance"])
        elif np.linalg.norm(residual) <= settings["rank_tolerance"] * np.linalg.norm(values):
            # An exactly explained factor can leave QR roundoff with arbitrary
            # ranks. Remove only numerical zero; do not rescale other residuals
            # or alter a raw signal, including genuinely tiny-valued inputs.
            residual = np.zeros_like(residual)
        return residual, info

    def append_signal(entry, group, variant, values, method, info):
        out = group.copy()
        out["variant"], out["signal"] = variant, values
        out["percentile"] = out.signal.rank(method="average", pct=True)
        out["group"] = np.ceil(10 * out.percentile).clip(1, 10).astype(int)
        signals.append(out)
        constant = np.ptp(values) == 0
        diagnostics.append({"entry": entry, "variant": variant,
                            "status": "constant" if constant else "ok", "reason": "constant_signal" if constant else "",
                            "n_stocks": len(group), "winsor_method": method, **info})

    for entry, group in frame.groupby("entry", sort=True):
        group = group.sort_values("code").copy()
        if sample_policy == "independent":
            finite_factor = np.isfinite(group.value.to_numpy(dtype=float))
            for variant in variants:
                use_cap = variant in {"size", "industry_size"} or (variant == "full" and include_float_mcap)
                use_industry = variant in {"industry", "industry_size", "full"}
                numeric = ([market_cap_column] if use_cap else []) + (RISK_FIELDS if variant == "full" else [])
                columns = numeric + (["industry"] if use_industry else [])
                missing = set(columns) - set(group)
                if missing:
                    diagnostics.append({"entry": entry, "variant": variant, "status": "failed",
                                        "reason": "missing_control_columns:" + ",".join(sorted(missing)),
                                        "n_stocks": 0, "winsor_method": "not_applied"})
                    continue
                valid = finite_factor.copy()
                if numeric:
                    valid &= np.isfinite(group[numeric].to_numpy(dtype=float)).all(axis=1)
                if use_cap:
                    valid &= group[market_cap_column].gt(0).to_numpy()
                if use_industry:
                    valid &= group.industry.astype("string").str.strip().ne("").fillna(False).to_numpy(dtype=bool)
                selected = group.loc[valid].copy()
                if selected.empty:
                    diagnostics.append({"entry": entry, "variant": variant, "status": "empty",
                                        "reason": ("no_valid_factor_values" if not finite_factor.any()
                                                   else "no_valid_control_observations"),
                                        "n_stocks": 0, "winsor_method": "not_applied"})
                    continue
                values, method = transform_base(selected.value.to_numpy())
                info = {"rank": 0, "columns": 0, "redundant_columns": ""}
                if variant != "raw":
                    try:
                        values, info = transform_neutral(values, selected, variant)
                    except ValueError as exc:
                        diagnostics.append({"entry": entry, "variant": variant, "status": "failed",
                                            "reason": str(exc), "n_stocks": len(selected), "winsor_method": method})
                        continue
                append_signal(entry, selected, variant, values, method, info)
            continue
        numeric = group[numeric_columns].to_numpy(dtype=float)
        if (not np.isfinite(numeric).all() or (needs_industry and group.industry.isna().any())
                or (needs_cap and group[market_cap_column].le(0).any())):
            raise ValueError("prepare_signals requires a frozen complete common sample")
        raw, method = transform_base(group.value.to_numpy())
        transformed = {"raw": raw}
        infos = {"raw": {"rank": 0, "columns": 0, "redundant_columns": ""}}
        failed = None
        for variant in (v for v in variants if v != "raw"):
            try:
                transformed[variant], infos[variant] = transform_neutral(raw, group, variant)
            except ValueError as exc:
                failed = str(exc)
                break
        if failed:
            diagnostics.append({"entry": entry, "variant": "all", "status": "failed",
                                "reason": failed, "n_stocks": len(group), "winsor_method": method})
            continue
        for variant in variants:
            append_signal(entry, group, variant, transformed[variant], method, infos[variant])
    signal_columns = list(frame.columns) + ["variant", "signal", "percentile", "group"]
    diag_columns = ["entry", "variant", "status", "reason", "n_stocks", "winsor_method",
                    "rank", "columns", "redundant_columns"]
    return (pd.concat(signals, ignore_index=True) if signals else pd.DataFrame(columns=signal_columns),
            pd.DataFrame(diagnostics, columns=diag_columns))
