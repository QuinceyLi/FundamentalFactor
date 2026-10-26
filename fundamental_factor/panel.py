from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .audit import audit, source_columns
from .config import Config, DataGateError, RISK_FIELDS
from .data import asof_snapshot, columns, fingerprint, read_frame, write_json
from .preprocess import prepare_signals


def _status_frame(config: Config) -> pd.DataFrame:
    """Read optional trading flags without turning absent observations into false."""
    path = config.source("status")
    classic = config.data_policy["mode"] == "classic"
    if classic and (path is None or not path.is_file()):
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                             "code": pd.Series(dtype="string"),
                             "suspended": pd.Series(dtype="boolean")})
    available = columns(path)
    selected = ["date", "code"] + [name for name in ("suspended", "limit_up", "limit_down")
                                    if name in available]
    status = read_frame(path, selected, nullable_booleans=True)
    if "suspended" not in status:
        if not classic:
            raise ValueError("status missing suspended column")
        status["suspended"] = pd.Series(pd.NA, index=status.index, dtype="boolean")
    return status


def weekly_calendar(calendar: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                    study: dict | None = None) -> pd.DataFrame:
    days = calendar.sort_values().drop_duplicates()
    # Group the full calendar first: slicing midweek must not invent a rebalance day.
    first = days.groupby(days.dt.to_period("W-SUN")).min().sort_values().reset_index(drop=True)
    first = first[(first >= pd.Timestamp(start)) & (first <= pd.Timestamp(end))].reset_index(drop=True)
    weeks = pd.DataFrame({"entry": first.iloc[:-1].to_numpy(), "exit": first.iloc[1:].to_numpy()})
    weeks["cutoff"] = weeks.entry - pd.Timedelta(nanoseconds=1)
    if weeks.empty:
        raise DataGateError("没有完整的周度开盘到开盘持有期")
    return weeks


def base_universe(config: Config, weeks: pd.DataFrame) -> pd.DataFrame:
    required = source_columns(config)
    policy = config.data_policy
    relaxed = policy["mode"] in ("proxy", "classic")
    stock_path = config.source("stocks")
    selected = required["stocks"] + (["is_a_share"] if "is_a_share" in columns(stock_path) else [])
    stocks = read_frame(stock_path, selected)
    if "is_a_share" in stocks:
        stocks = stocks[stocks.is_a_share]
    if config.exchanges is not None:
        stocks = stocks[stocks.code.str.rsplit(".", n=1).str[-1].isin(config.exchanges)]
    calendar = read_frame(config.source("calendar"), ["date"]).date.sort_values().to_numpy()
    grid = weeks.merge(stocks, how="cross")
    grid = grid[(grid.listing_date <= grid.cutoff) &
                (grid.delisting_date.isna() | grid.delisting_date.gt(grid.cutoff))].copy()
    grid["listing_sessions"] = (np.searchsorted(calendar, grid.cutoff.to_numpy(), side="right") -
                                np.searchsorted(calendar, grid.listing_date.to_numpy(), side="left"))
    history_path = config.source("history")
    if policy["mode"] == "classic" and (history_path is None or not history_path.is_file()):
        grid["history_date"] = pd.NaT
        grid["is_st"] = pd.Series(pd.NA, index=grid.index, dtype="boolean")
    else:
        history_columns = required.get("history", ["date", "code"])
        available = columns(history_path)
        if policy["mode"] == "classic":
            history_columns = [name for name in history_columns if name != "is_st" or name in available]
            # ST remains an optional stock-pool input even in a raw-only study.
            if "is_st" in available and "is_st" not in history_columns:
                history_columns = history_columns + ["is_st"]
        history = read_frame(history_path, history_columns, nullable_booleans=relaxed)
        if "is_st" not in history:
            history["is_st"] = pd.Series(pd.NA, index=history.index, dtype="boolean")
        grid = asof_snapshot(grid, history, "history")
    if "industry" not in grid:
        grid["industry"] = pd.NA
    grid["short_listing"] = grid.listing_sessions.lt(config.raw["backtest"]["min_listing_sessions"])
    # A proxy study may retain unknown ST states, but never relabel them clean.
    unresolved_st = grid.is_st.isna() & ~grid.short_listing
    grid["is_st_unknown"] = grid.is_st.isna()
    if unresolved_st.any() and policy["st_unknown"] == "error":
        raise DataGateError(f"{int(unresolved_st.sum())} 条股票周次缺少截点历史ST状态")
    status = _status_frame(config)[["date", "code", "suspended"]]
    grid = grid.merge(status.rename(columns={"date": "entry", "suspended": "entry_suspended"}),
                      on=["entry", "code"], how="left", validate="many_to_one")
    st_excluded = grid.is_st.astype("boolean").fillna(policy["st_unknown"] != "include").astype(bool)
    relevant = ~grid.short_listing & ~st_excluded
    grid["entry_suspension_unknown"] = grid.entry_suspended.isna()
    if (grid.entry_suspension_unknown & relevant).any() and policy["suspension_unknown"] == "error":
        raise DataGateError("基础股票池存在缺失的入场日停牌状态")
    suspension_excluded = grid.entry_suspended.astype("boolean").fillna(
        policy["suspension_unknown"] != "include").astype(bool)
    grid["eligible"] = relevant & ~suspension_excluded
    grid.to_parquet(config.output / "universe.parquet", index=False)
    return grid[grid.eligible].reset_index(drop=True)


def signal_panel(config: Config, weeks: pd.DataFrame, base: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    include_float_mcap = config.include_float_mcap
    base = base_universe(config, weeks) if base is None else base
    if config.requires_risk:
        risk = read_frame(config.source("risk"), ["date", "code"] + RISK_FIELDS)
        controls = asof_snapshot(base, risk, "risk")
        del risk
        controls[RISK_FIELDS] = controls[RISK_FIELDS].replace([np.inf, -np.inf], np.nan)
    else:
        controls = base.copy()
        controls["risk_date"] = pd.NaT
    controls["industry"] = controls.industry.replace(r"^\s*$", np.nan, regex=True)
    numeric_columns = (["float_mcap"] if config.requires_float_mcap else []) + (RISK_FIELDS if config.requires_risk else [])
    numeric_ok = np.isfinite(controls[numeric_columns].to_numpy(dtype=float)).all(axis=1)
    controls["controls_ok"] = numeric_ok
    if config.requires_industry:
        controls["controls_ok"] &= controls.industry.notna()
    if config.requires_float_mcap:
        controls["controls_ok"] &= controls.float_mcap.gt(0)
    controls.to_parquet(config.output / "controls.parquet", index=False)
    blocks, coverage, failures = [], [], []
    for factor in config.factors:
        source = read_frame(config.source(factor["source"]), ["date", "code", factor["column"]])
        source = source.rename(columns={factor["column"]: "value"})
        source["value"] = source.value.replace([np.inf, -np.inf], np.nan)
        joined = asof_snapshot(controls, source, "factor")
        del source
        joined["factor"] = factor["name"]
        joined["factor_ok"] = joined.value.notna()
        joined["common_ok"] = joined.factor_ok & joined.controls_ok
        counts = joined.groupby("entry").agg(base_count=("code", "size"),
                    factor_available=("factor_ok", "sum"), common_count=("common_ok", "sum"),
                    unknown_st_count=("is_st_unknown", "sum"),
                    unknown_suspension_count=("entry_suspension_unknown", "sum"))
        counts = counts.reindex(weeks.entry, fill_value=0)
        counts["factor"] = factor["name"]
        counts["missing_factor"] = counts.base_count - counts.factor_available
        counts["missing_controls"] = counts.factor_available - counts.common_count
        denominator = counts.base_count.replace(0, np.nan)
        counts["raw_coverage"] = counts.factor_available / denominator
        counts["common_coverage"] = counts.common_count / denominator
        counts["unknown_st_fraction"] = counts.unknown_st_count / denominator
        counts["unknown_suspension_fraction"] = counts.unknown_suspension_count / denominator
        common = joined[joined.common_ok].copy()
        zero = common.value.eq(0).groupby(common.entry).mean()
        ties = common.groupby("entry").value.apply(lambda s: s.duplicated(keep=False).mean())
        counts["zero_fraction"] = zero.reindex(counts.index)
        counts["tie_fraction"] = ties.reindex(counts.index)
        coverage.append(counts.reset_index())
        signals, diagnostics = prepare_signals(common, config.raw["preprocess"],
                                               include_float_mcap=include_float_mcap,
                                               variants=config.variants)
        diagnostics["factor"] = factor["name"]
        failures.append(diagnostics)
        # Controls are cached once; avoid writing 20 exposures six times for two factors.
        keep = ["entry", "exit", "cutoff", "code", "factor", "value", "factor_date", "history_date", "risk_date",
                "variant", "signal", "percentile", "group", "industry", "is_st_unknown", "entry_suspension_unknown"]
        if config.requires_float_mcap:
            keep.append("float_mcap")
        blocks.append(signals[keep])
    return pd.concat(blocks, ignore_index=True), pd.concat(coverage, ignore_index=True), pd.concat(failures, ignore_index=True)


def make_labels(config: Config, features: pd.DataFrame) -> pd.DataFrame:
    required = source_columns(config)
    price = config.raw["price_contract"]
    classic = config.data_policy["mode"] == "classic"
    quotes = read_frame(config.source("quotes"), required["quotes"])
    quotes = quotes.rename(columns={price.get("open_column", "acc_open"): "open_level",
                                    price.get("close_column", "acc_close"): "close_level"})
    status = _status_frame(config)
    base = features[["entry", "exit", "code", "delisting_date"]].drop_duplicates().copy().reset_index(drop=True)
    if base.duplicated(["entry", "exit", "code"]).any():
        raise ValueError("inconsistent label metadata")
    base["row_id"] = np.arange(len(base))
    for endpoint in ("entry", "exit"):
        target = base[["row_id", "code", endpoint]].rename(columns={endpoint: "date"})
        point = target.merge(quotes[["date", "code", "open_level"]], on=["date", "code"], how="left", validate="many_to_one")
        point = point.merge(status, on=["date", "code"], how="left", validate="many_to_one")
        point["cutoff"] = point.date - pd.Timedelta(nanoseconds=1)
        previous = asof_snapshot(point[["row_id", "code", "cutoff"]],
                                 quotes[["date", "code", "close_level"]], "valuation")
        point = point.merge(previous[["row_id", "close_level"]], on="row_id", validate="one_to_one")
        suspended = point.suspended.astype("boolean").fillna(False).astype(bool)
        point["level"] = point.open_level.where(~suspended, point.close_level)
        point["status_missing"] = point.suspended.isna()
        proxy_carry = (config.data_policy["mode"] in ("proxy", "classic") and
                       price.get("suspension_valuation", "previous_adjusted_close" if classic else None)
                       == "previous_adjusted_close")
        point["valuation_proxy"] = suspended & proxy_carry & (not bool(price.get("suspended_total_return", False)))
        point["valuation_unverified"] = suspended & (not (
            bool(price.get("suspended_total_return", False)) or proxy_carry))
        keep = ["row_id", "level", "suspended", "status_missing", "valuation_unverified", "valuation_proxy"] + [c for c in ("limit_up", "limit_down") if c in point]
        base = base.merge(point[keep].rename(columns={c: f"{endpoint}_{c}" for c in keep if c != "row_id"}),
                          on="row_id", validate="one_to_one")
    base["label_ready"] = base.exit
    dead = base.delisting_date.notna() & base.delisting_date.le(base.exit)
    base["terminal_applied"] = False
    terminal_path = config.source("terminal")
    if dead.any() and terminal_path and terminal_path.is_file():
        terminal = read_frame(terminal_path, required["terminal"])
        events = base.loc[dead, ["row_id", "code", "delisting_date"]].merge(
            terminal.rename(columns={"date": "delisting_date"}), on=["code", "delisting_date"],
            how="left", validate="many_to_one").set_index("row_id")
        # A valid empty CSV has object-typed numeric columns; its unmatched
        # events are missing values, not terminal payments or a schema failure.
        events["terminal_value"] = pd.to_numeric(events.terminal_value, errors="raise")
        valid = events.terminal_value.notna() & np.isfinite(events.terminal_value) & events.terminal_value.ge(0)
        ids = events.index[valid]
        base.loc[ids, "exit_level"] = events.loc[ids, "terminal_value"].to_numpy()
        base.loc[ids, "terminal_applied"] = True
        base.loc[ids, "label_ready"] = pd.concat([base.loc[ids, "exit"].reset_index(drop=True),
                events.loc[ids, "available_date"].reset_index(drop=True)], axis=1).max(axis=1).to_numpy()
    base["delisting_without_terminal"] = dead & ~base.terminal_applied
    # A carried pre-delisting mark is not an observed exit price or a settlement.
    missing_delisting_exit = base.delisting_without_terminal & (
        (not classic) | base.exit_suspended.astype("boolean").fillna(False).astype(bool))
    invalid_exit = ~np.isfinite(base.exit_level) | base.exit_level.lt(0)
    if classic:
        # A vendor zero price is missing market data; only explicit terminal
        # evidence may represent a complete loss of the investment.
        invalid_exit |= base.exit_level.eq(0) & ~base.terminal_applied
    base["unresolved"] = ((base.entry_status_missing & (not classic)) | base.entry_valuation_unverified |
        (((base.exit_status_missing & (not classic)) | base.exit_valuation_unverified) & ~base.terminal_applied) |
        missing_delisting_exit | ~np.isfinite(base.entry_level) | invalid_exit | base.entry_level.le(0))
    base["return"] = (base.exit_level / base.entry_level - 1).where(~base.unresolved)
    base["unresolved_reason"] = ""
    reasons = {
        "entry_status_missing": base.entry_status_missing & (not classic),
        "entry_valuation_unverified": base.entry_valuation_unverified,
        "exit_status_missing": base.exit_status_missing & ~base.terminal_applied & (not classic),
        "exit_valuation_unverified": base.exit_valuation_unverified & ~base.terminal_applied,
        "delisting_without_terminal": base.delisting_without_terminal & base.unresolved,
        "invalid_entry_price": ~np.isfinite(base.entry_level) | base.entry_level.le(0),
        "invalid_exit_price": invalid_exit,
    }
    for reason, mask in reasons.items():
        base.loc[mask, "unresolved_reason"] += reason + ";"
    base["unresolved_reason"] = base.unresolved_reason.str.rstrip(";")
    base["trade_restricted"] = base.terminal_applied | dead
    base["trade_restriction_unknown"] = base.entry_status_missing | base.exit_status_missing
    for col in ("entry_suspended", "exit_suspended", "entry_limit_up", "entry_limit_down", "exit_limit_up", "exit_limit_down"):
        if col in base:
            base["trade_restricted"] |= base[col].astype("boolean").fillna(False).astype(bool)
            base["trade_restriction_unknown"] |= base[col].isna()
        elif "limit_" in col:
            base["trade_restriction_unknown"] = True
    return base.drop(columns=["row_id", "delisting_date"])


def build_panel(config: Config) -> dict:
    if (config.output / "results.json").exists():
        raise DataGateError("该目录已有收益结果；重建面板请使用新的 output_dir")
    checked = audit(config)
    if not checked["gate_passed"]:
        raise DataGateError("正式回测数据门槛未通过；见 audit.json")
    quote = checked["sources"]["quotes"]
    factor_start = min(pd.Timestamp(checked["sources"][f["source"]]["start"]) for f in config.factors)
    start = max(pd.Timestamp(quote["start"]), factor_start + pd.Timedelta(days=1))
    end = pd.Timestamp(quote["end"])
    settings = config.raw["backtest"]
    if settings["start_date"]:
        start = max(start, pd.Timestamp(settings["start_date"]))
    if settings["end_date"]:
        end = min(end, pd.Timestamp(settings["end_date"]))
    calendar = read_frame(config.source("calendar"), ["date"])
    weeks = weekly_calendar(calendar.date, start, end)
    execution = config.raw.get("execution", {})
    if (config.data_policy["mode"] != "classic" and execution.get("enabled")
            and "coverage_start" in execution and (
            pd.Timestamp(execution["coverage_start"]) > weeks.entry.min()
            or pd.Timestamp(execution["coverage_end"]) < weeks.exit.max())):
        raise DataGateError("执行账本已核实覆盖期不足完整持有期；不可用缺失事件冒充无公司行动")
    weeks.to_parquet(config.output / "weeks.parquet", index=False)
    base = base_universe(config, weeks)
    if base.empty:
        raise DataGateError("过滤后基础股票池为空")
    features, coverage, diagnostics = signal_panel(config, weeks, base)
    coverage.to_csv(config.output / "coverage.csv", index=False)
    diagnostics.to_csv(config.output / "preprocessing_failures.csv", index=False)
    features.to_parquet(config.output / "features.parquet", index=False)
    manifest = {"schema_version": 2, "calculation_version": 3,
                "config_digest": config.digest, "status": "features_frozen",
                "weeks": len(weeks), "signal_rows": len(features), "base_rows": len(base),
                "unknown_st_base_rows": int(base.is_st_unknown.sum()),
                "unknown_suspension_base_rows": int(base.entry_suspension_unknown.sum()),
                "signal_start": weeks.cutoff.min(), "signal_end": weeks.cutoff.max(),
                "holding_start": weeks.entry.min(), "holding_end": weeks.exit.max(),
                "source_fingerprints": {n: d["fingerprint"] for n, d in checked["sources"].items()}}
    write_json(config.output / "panel_manifest.json", manifest)
    # Features and group assignments are on disk before forward returns are connected.
    labels = make_labels(config, base)
    labels.to_parquet(config.output / "labels.parquet", index=False)
    unresolved = labels[labels.unresolved]
    unresolved.to_csv(config.output / "unresolved_labels.csv", index=False)
    manifest["unresolved_labels"] = len(unresolved)
    manifest["label_coverage"] = float((~labels.unresolved).mean()) if len(labels) else 0.
    manifest["unknown_trade_restriction_rows"] = int(labels.trade_restriction_unknown.sum())
    manifest["delisting_without_terminal_rows"] = int(labels.delisting_without_terminal.sum())
    manifest["priced_delisting_without_terminal_rows"] = int((labels.delisting_without_terminal & ~labels.unresolved).sum())
    manifest["suspension_valuation_proxy_rows"] = int((labels.entry_valuation_proxy | labels.exit_valuation_proxy).sum())
    manifest["status"] = "unresolved_labels" if len(unresolved) else "ready"
    rows = []
    for week in weeks.itertuples(index=False):
        group = labels[labels.entry.eq(week.entry)]
        n_labels = int((~group.unresolved).sum())
        data_complete = len(group) > 0 and not group.unresolved.any()
        available_case = config.data_policy["missing_returns"] == "available_case"
        ready = n_labels > 0 if available_case else data_complete
        rows.append({"entry": week.entry, "exit": week.exit, "n_stocks": len(group),
                     "n_labels": n_labels,
                     "label_coverage": float((~group.unresolved).mean()) if len(group) else 0.,
                     "return": float(group["return"].mean()) if ready else np.nan,
                     "data_complete": data_complete, "complete": ready,
                     "missing_weight": float(group.unresolved.mean()) if len(group) else np.nan,
                     "return_method": ("full_sample" if data_complete else "available_case" if ready
                                       else "missing_return"),
                     "trade_restricted_fraction": float(group.trade_restricted.mean()) if len(group) else np.nan,
                     "trade_restriction_unknown_fraction": float(group.trade_restriction_unknown.mean()) if len(group) else np.nan})
    benchmark = pd.DataFrame(rows)
    benchmark.to_csv(config.output / "benchmark.csv", index=False)
    if benchmark["return"].isna().any() and manifest["status"] == "ready":
        manifest["status"] = "empty_benchmark"
    manifest["incomplete_benchmark_weeks"] = int(benchmark["return"].isna().sum())
    manifest["partial_benchmark_weeks"] = int((benchmark["return"].notna() & ~benchmark.data_complete).sum())
    manifest["data_policy"] = config.data_policy
    if config.data_policy["missing_returns"] in ("report", "available_case"):
        manifest["quality_status"] = manifest["status"]
        manifest["status"] = "ready"
    files = ["weeks.parquet", "universe.parquet", "controls.parquet", "features.parquet", "labels.parquet",
             "benchmark.csv", "coverage.csv", "preprocessing_failures.csv"]
    manifest["artifact_fingerprints"] = {n: fingerprint(config.output / n) for n in files}
    write_json(config.output / "panel_manifest.json", manifest)
    if manifest["status"] != "ready":
        raise DataGateError(f"面板状态 {manifest['status']}；{len(unresolved)} 条收益或状态未解决")
    return manifest


def ensure_panel(config: Config) -> dict:
    path = config.output / "panel_manifest.json"
    if not path.is_file():
        return build_panel(config)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("config_digest") != config.digest:
        raise DataGateError("冻结配置或面板版本不一致；请使用新的 output_dir")
    if config.data_policy["mode"] == "classic" and manifest.get("calculation_version") != 3:
        raise DataGateError("旧面板不含经典因子统计口径版本；请使用新的 output_dir 重建")
    for name, old in manifest["source_fingerprints"].items():
        source = config.source(name)
        if not source or not source.is_file() or fingerprint(source) != old:
            raise DataGateError(f"源文件 {name} 已改变；请新建输出目录重新构建")
    if manifest["status"] != "ready":
        raise DataGateError(f"面板未通过门槛：{manifest['status']}")
    for name, old in manifest["artifact_fingerprints"].items():
        target = config.output / name
        if not target.is_file() or fingerprint(target) != old:
            raise DataGateError(f"冻结面板产物已改变：{name}")
    return manifest
