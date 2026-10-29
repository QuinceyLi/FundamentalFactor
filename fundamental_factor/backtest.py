"""Weekly factor tests with frozen signal targets and auditable return weights.

Returns are gross statistical portfolio returns.  The position ledger is the
source of turnover: it records drifted old positions, targets, and exits instead
of approximating turnover from the fraction of changed group members.
Available-case returns use separately recorded weights over observed labels;
missing old returns never silently become zero in the target-position ledger.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import DataGateError
from .inference import mean_interval


SUBSETS = ("ALL", "TOP20", "TOP50", "BOTTOM50", "BOTTOM20")
PORTFOLIOS = ("ALL", *(f"Q{i}" for i in range(1, 11)), "Q10-Q1")


def rank_ic(score, target, method: str = "spearman") -> float:
    """Paired finite observations, at least three stocks, and nonconstant inputs.

    The caller preserves frozen signal membership and reports label coverage.
    Available-case group returns record their separate evaluation weights.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError("IC method must be spearman or pearson")
    x, y = np.asarray(score, dtype=float), np.asarray(target, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("IC inputs must be one-dimensional and equally sized")
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    if method == "spearman":
        x, y = pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average")
    return float(np.corrcoef(x, y)[0, 1])


def drift_turnover(target: Mapping[str, float],
                   previous: Mapping[str, float] | None = None,
                   previous_returns: Mapping[str, float] | None = None
                   ) -> tuple[float, dict[str, float]]:
    """Return double-sided turnover and pre-trade weights.

    Target and previous weights are fully invested, nonnegative long portfolios
    (or empty).  The first investment therefore has turnover one; exchanging all
    holdings has turnover two.  No terminal liquidation is added.
    """
    target = {str(k): float(v) for k, v in target.items()}
    previous = None if previous is None else {str(k): float(v) for k, v in previous.items()}
    for name, weights in (("target", target), ("previous", previous or {})):
        if any(not np.isfinite(w) or w < 0 for w in weights.values()):
            raise ValueError(f"{name} weights must be finite and nonnegative")
        if weights and not np.isclose(sum(weights.values()), 1.0, atol=1e-10):
            raise ValueError(f"{name} weights must sum to one")
    if not previous:
        drifted = {}
    else:
        if previous_returns is None or any(code not in previous_returns for code in previous):
            raise DataGateError("Previous held-stock returns are required to compute turnover")
        r = {code: float(previous_returns[code]) for code in previous}
        if any(not np.isfinite(value) or value < -1 for value in r.values()):
            raise DataGateError("Invalid previous held-stock return in turnover ledger")
        denominator = sum(previous[code] * (1 + r[code]) for code in previous)
        if denominator <= 0:
            raise DataGateError("Bankrupt previous portfolio has no defined drifted weights")
        drifted = {code: previous[code] * (1 + r[code]) / denominator for code in previous}
    turnover = sum(abs(target.get(code, 0.0) - drifted.get(code, 0.0))
                   for code in target.keys() | drifted.keys())
    return float(turnover), drifted


def _sharpe_stats(returns, annualization: int) -> tuple[float, int]:
    """Arithmetic mean / sample standard deviation, on finite observations."""
    values = np.asarray(returns, dtype=float)
    valid = values[np.isfinite(values)]
    n = len(valid)
    sd = float(valid.std(ddof=1)) if n > 1 else np.nan
    value = (float(valid.mean() / sd * np.sqrt(annualization))
             if n > 1 and np.ptp(valid) > 0 and np.isfinite(sd) and sd > 0 else np.nan)
    return value, n


def _return_metrics(returns, annualization: int) -> dict:
    series = np.asarray(returns, dtype=float)
    if series.ndim != 1:
        raise ValueError("Returns must be a one-dimensional weekly series")
    failures = np.flatnonzero(np.isfinite(series) & (series <= -1))
    failure = int(failures[0]) if len(failures) else None
    # Once capital is zero there is no next-period investment to earn the
    # supplied later returns.  This also protects direct helper callers from
    # reporting a recovery after bankruptcy.
    observed = series if failure is None else series[:failure + 1]
    valid = observed[np.isfinite(observed)]
    n = len(valid)
    complete = bool(len(observed) > 0 and n == len(observed))
    bankrupt = failure is not None and series[failure] == -1
    insolvent = failure is not None and series[failure] < -1
    volatility = float(np.std(valid, ddof=1) * np.sqrt(annualization)) if n > 1 else np.nan
    sharpe, _ = _sharpe_stats(valid, annualization)
    result = {"n_periods": n, "total_return": np.nan, "annual_return": np.nan,
              "annual_volatility": volatility, "information_ratio": np.nan,
              "sharpe_ratio": sharpe, "sharpe_n_periods": n,
              "win_rate": float(np.mean(valid > 0)) if n else np.nan,
              "max_drawdown": np.nan,
              "status": ("insolvency" if insolvent else "bankruptcy" if bankrupt else
                         "ok" if complete else "incomplete")}
    if complete and bankrupt:
        # A complete path ending at zero is a valid -100% result, including
        # its drawdown.  It is not the invalid negative-equity case below -1.
        result.update(total_return=-1.0, annual_return=-1.0, max_drawdown=1.0,
                      information_ratio=-1.0 / volatility if volatility > 0 else np.nan)
    elif complete and not insolvent:
        log_nav = np.cumsum(np.log1p(series))
        if not np.isfinite(log_nav).all() or log_nav.max() > 700:
            result["status"] = "numerical_overflow"
            return result
        nav = np.concatenate(([1.0], np.exp(log_nav)))
        annual_log = log_nav[-1] * annualization / len(series)
        if annual_log > 700:
            result["status"] = "numerical_overflow"
            return result
        annual = float(np.expm1(annual_log))
        result.update(total_return=float(nav[-1] - 1), annual_return=annual,
                      information_ratio=annual / volatility if volatility > 0 else np.nan,
                      max_drawdown=float(np.max(1 - nav / np.maximum.accumulate(nav))))
    return result


def _turnover_diagnostics(turnover, shape, initial_build, threshold) -> dict:
    """Observe continuing rebalances, excluding first funding of each leg.

    One-way turnover is half the existing double-sided position change. For
    Q10-Q1 the two unit-notional legs are summed before dividing by two; this
    is not turnover divided by the spread's two-unit gross exposure.
    """
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold) or threshold < 0):
        raise ValueError("turnover_warning_threshold must be finite and nonnegative")
    initial = np.zeros(shape, dtype=bool) if initial_build is None else np.asarray(initial_build, dtype=bool)
    if initial.shape != shape:
        raise ValueError("Returns and initial-build flags must share the same period grid")
    result = {"mean_turnover_one_way": np.nan, "p95_turnover_one_way": np.nan,
              "high_turnover_fraction": np.nan, "turnover_observations": 0,
              "turnover_missing_periods": 0, "initial_build_periods": int(initial.sum()),
              "turnover_warning_threshold": float(threshold)}
    if turnover is None:
        return result
    traded = np.asarray(turnover, dtype=float)
    if traded.shape != shape:
        raise ValueError("Returns and turnover must share the same period grid")
    if (traded[np.isfinite(traded)] < 0).any():
        raise ValueError("Turnover must be nonnegative when observed")
    continuing = traded[~initial] / 2.
    good = continuing[np.isfinite(continuing)]
    result.update(turnover_observations=len(good),
                  turnover_missing_periods=int((~np.isfinite(continuing)).sum()))
    if len(good):
        result.update(mean_turnover_one_way=float(good.mean()),
                      p95_turnover_one_way=float(np.quantile(good, .95)),
                      high_turnover_fraction=float(np.mean(good > threshold)))
    return result


def performance_metrics(returns, turnover=None, excess_returns=None,
                        annualization: int = 52, target_turnover=None,
                        initial_build=None, turnover_warning_threshold: float = .5,
                        entries=None, exits=None) -> dict:
    """Attachment-compatible CAGR/volatility statistics on a full weekly grid.

    Gaps are never compressed for cumulative results.  Valid-period volatility
    and win rate remain visible, but total/CAGR/drawdown and annual turnover are
    undefined whenever a full continuous investment history is unavailable.
    """
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    values = np.asarray(returns, dtype=float)
    breaks = _calendar_breaks(len(values), entries, exits)
    result = _return_metrics(_values_with_gaps(values, breaks), annualization)
    result.update(_turnover_diagnostics(turnover, values.shape, initial_build,
                                       turnover_warning_threshold))
    result["annual_turnover"] = np.nan
    result["annual_target_turnover"] = np.nan
    if turnover is not None:
        traded = np.asarray(turnover, dtype=float)
        if traded.shape != values.shape:
            raise ValueError("Returns and turnover must share the same weekly grid")
        if result["status"] == "ok" and np.isfinite(traded).all():
            result["annual_turnover"] = float(traded.mean() * annualization)
    if target_turnover is not None:
        target_traded = np.asarray(target_turnover, dtype=float)
        if target_traded.shape != values.shape:
            raise ValueError("Returns and target turnover must share the same weekly grid")
        # This describes changes to the signal's target weights, independently
        # of realized price drift. A missing target cannot be inferred as cash.
        if len(target_traded) and np.isfinite(target_traded).all() and not breaks.any():
            result["annual_target_turnover"] = float(target_traded.mean() * annualization)
    excess = (_return_metrics(_values_with_gaps(excess_returns, breaks), annualization) if excess_returns is not None
              else {key: np.nan for key in _return_metrics([], annualization)})
    if excess_returns is not None and np.asarray(excess_returns).shape != values.shape:
        raise ValueError("Returns and excess returns must share the same weekly grid")
    for key in ("total_return", "annual_return", "annual_volatility", "information_ratio",
                "win_rate", "max_drawdown"):
        result[f"excess_{key}"] = excess[key]
    result["excess_sharpe"] = np.nan
    result["excess_sharpe_n_periods"] = 0
    if excess_returns is not None:
        # A benchmark-relative return can be below -100% without destroying
        # the actual long portfolio. Do not apply the legacy excess NAV's
        # bankruptcy truncation to the new arithmetic Sharpe observations.
        failures = np.flatnonzero(np.isfinite(values) & (values <= -1))
        stop = int(failures[0]) + 1 if len(failures) else len(values)
        result["excess_sharpe"], result["excess_sharpe_n_periods"] = _sharpe_stats(
            np.asarray(excess_returns, dtype=float)[:stop], annualization)
    if excess_returns is not None and excess["status"] in {"bankruptcy", "insolvency"}:
        result["status"] += f";excess_{excess['status']}"
    return result


def _calendar_breaks(length: int, entries=None, exits=None) -> np.ndarray:
    """Locate discontinuities without assuming that a month has a fixed length."""
    result = np.zeros(length, dtype=bool)
    if entries is None and exits is None:
        return result
    if entries is None or exits is None or len(entries) != length or len(exits) != length:
        raise ValueError("Entry and exit dates must share the return period grid")
    entry_dates, exit_dates = pd.to_datetime(entries), pd.to_datetime(exits)
    if entry_dates.isna().any() or exit_dates.isna().any():
        raise ValueError("Entry and exit dates must not contain missing dates")
    result[1:] = np.asarray(entry_dates)[1:] != np.asarray(exit_dates)[:-1]
    return result


def _values_with_gaps(values, breaks: np.ndarray) -> np.ndarray:
    """Keep absent calendar periods visible to cumulative statistics and HAC."""
    values = np.asarray(values, dtype=float)
    if values.shape != breaks.shape:
        raise ValueError("Values and calendar breaks must share the return period grid")
    return np.insert(values, np.flatnonzero(breaks), np.nan) if breaks.any() else values


def _segmented_path(values, compound: bool = True, *, entries=None, exits=None
                    ) -> tuple[list[float], list[int | None]]:
    """Rebase valid runs after gaps; never connect an invalid gap with a line."""
    path, ids, running, segment, active, bankrupt = [], [], 1.0 if compound else 0.0, 0, False, False
    terminal_value = np.nan
    values = np.asarray(values, dtype=float)
    breaks = _calendar_breaks(len(values), entries, exits)
    for value, discontinuous in zip(values, breaks):
        if bankrupt:
            path.append(terminal_value)
            ids.append(segment if np.isfinite(terminal_value) else None)
            continue
        if discontinuous:
            active = False
        if not np.isfinite(value):
            path.append(np.nan)
            ids.append(None)
            active = False
            continue
        if not active:
            segment += 1
            running = 1.0 if compound else 0.0
            active = True
        if compound and value <= -1:
            terminal_value = 0.0 if value == -1 else np.nan
            path.append(terminal_value)
            ids.append(segment)
            bankrupt = True
            continue
        running = running * (1 + value) if compound else running + value
        path.append(float(running))
        ids.append(segment)
    return path, ids


def _prepare_inputs(signals: pd.DataFrame, labels: pd.DataFrame, benchmark: pd.DataFrame,
                    weeks: pd.DataFrame, missing_returns: str = "error", sample_policy: str = "common"
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sample_policy not in ("common", "independent"):
        raise ValueError("sample_policy must be common or independent")
    if missing_returns not in ("error", "report", "available_case"):
        raise ValueError("missing_returns must be error, report or available_case")
    allow_missing = missing_returns != "error"
    inputs = []
    for name, frame, required, keys in (
        ("signals", signals, {"entry", "exit", "code", "variant", "signal", "percentile", "group"},
         ["entry", "exit", "variant", "code"]),
        ("labels", labels, {"entry", "exit", "code", "return"}, ["entry", "exit", "code"]),
        ("benchmark", benchmark, {"entry", "exit", "return"}, ["entry", "exit"]),
        ("weeks", weeks, {"entry", "exit"}, ["entry", "exit"]),
    ):
        if not required.issubset(frame):
            raise ValueError(f"{name} missing fields: {sorted(required - set(frame))}")
        frame = frame.copy()
        for column in ("entry", "exit"):
            frame[column] = pd.to_datetime(frame[column])
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name} contains null/duplicate keys")
        if (frame.exit <= frame.entry).any():
            raise ValueError(f"{name} contains a nonpositive holding interval")
        inputs.append(frame)
    signals, labels, benchmark, weeks = inputs
    weeks = weeks.sort_values("entry").reset_index(drop=True)
    if weeks.empty or weeks.entry.duplicated().any():
        raise ValueError("A nonempty unique weekly grid is required")
    if len(weeks) > 1 and (weeks.exit.iloc[:-1].to_numpy() > weeks.entry.iloc[1:].to_numpy()).any():
        raise ValueError("Weekly holding intervals overlap")
    if signals.empty:
        raise DataGateError("No common factor/control observations are available")
    for name in ("signal", "percentile", "group"):
        if not np.isfinite(signals[name].to_numpy(dtype=float)).all():
            raise DataGateError(f"Nonfinite {name} in frozen factor sample")
    if (signals.percentile <= 0).any() or (signals.percentile > 1).any():
        raise ValueError("Factor percentiles must be in (0, 1]")
    if not signals.group.isin(range(1, 11)).all():
        raise ValueError("Group labels must be integers 1 through 10")
    known = weeks[["entry", "exit"]].assign(_known=True)
    if signals.merge(known, on=["entry", "exit"], how="left")._known.isna().any():
        raise ValueError("Signal rows fall outside the requested weekly grid")
    bench = weeks[["entry", "exit"]].merge(
        benchmark, on=["entry", "exit"], how="left", validate="one_to_one")
    active_periods = bench.entry.isin(signals.entry)
    if not allow_missing and not np.isfinite(bench.loc[active_periods, "return"].to_numpy(dtype=float)).all():
        raise DataGateError("The independent base-universe benchmark has unresolved weekly returns")
    if "unresolved" in bench:
        unresolved_benchmark = bench.unresolved.astype("boolean").fillna(True)
        if not allow_missing and (unresolved_benchmark & active_periods).any():
            raise DataGateError("The independent base-universe benchmark has unresolved labels")
        if missing_returns != "available_case":
            bench.loc[unresolved_benchmark, "return"] = np.nan
    bench.loc[~np.isfinite(bench["return"]), "return"] = np.nan
    if (bench["return"] < -1).any():
        raise DataGateError("Benchmark returns below -100% are invalid")
    payload = ["entry", "exit", "code", "return"]
    payload += [col for col in ("unresolved", "trade_restricted", "trade_restriction_unknown") if col in labels]
    if "return" in signals:
        raise ValueError("Signal features must not already contain a future return column")
    joined = signals.merge(labels[payload], on=["entry", "exit", "code"], how="left",
                           validate="many_to_one", indicator="_label_join")
    bad = (joined._label_join != "both") | ~np.isfinite(joined["return"].to_numpy(dtype=float))
    if "unresolved" in joined:
        bad |= joined.unresolved.astype("boolean").fillna(True).astype(bool)
    if bad.any() and not allow_missing:
        examples = joined.loc[bad, ["entry", "code"]].head(5).astype(str).to_dict("records")
        raise DataGateError(f"Unresolved frozen held-stock labels ({int(bad.sum())} rows): {examples}")
    if (joined["return"] < -1).any():
        raise DataGateError("Held-stock returns below -100% are invalid")
    joined.loc[bad, "return"] = np.nan
    joined["label_available"] = ~bad
    joined = joined.drop(columns="_label_join")
    joined["trade_restricted"] = joined.get("trade_restricted", False)
    if not allow_missing and joined.trade_restricted.isna().any():
        raise DataGateError("Missing trade-restriction status in held-stock labels")
    joined["trade_restriction_unknown"] = joined.get("trade_restriction_unknown", False)
    joined["trade_restriction_unknown"] = (
        joined.trade_restriction_unknown.astype("boolean").fillna(True).astype(bool) |
        joined.trade_restricted.isna())
    variants = sorted(signals.variant.unique())
    if sample_policy == "common":
        for _, block in signals.groupby(["entry", "exit"], sort=False):
            sets = [set(block.loc[block.variant == variant, "code"]) for variant in variants]
            if any(stock_set != sets[0] for stock_set in sets[1:]):
                raise DataGateError("Preprocessing variants do not share the same frozen stock sample")
    return joined, labels, bench, weeks


def evaluate_factor(signals: pd.DataFrame, labels: pd.DataFrame, benchmark: pd.DataFrame,
                    weeks: pd.DataFrame, annualization: int = 52,
                    nw_lag: int = 13, missing_returns: str = "error",
                    empty_groups: str = "missing", sample_policy: str = "common",
                    variants: list[str] | None = None,
                    turnover_warning_threshold: float = .5) -> dict[str, pd.DataFrame]:
    """Evaluate frozen signals over supplied weekly or monthly holding intervals."""
    if empty_groups not in ("missing", "cash"):
        raise ValueError("empty_groups must be missing or cash")
    _turnover_diagnostics(None, (0,), None, turnover_warning_threshold)
    joined, labels, benchmark, weeks = _prepare_inputs(signals, labels, benchmark, weeks, missing_returns, sample_policy)
    available_case = missing_returns == "available_case"
    observed_variants = set(joined.variant.unique())
    variants = sorted(observed_variants) if variants is None else list(variants)
    if not variants or len(set(variants)) != len(variants) or not observed_variants.issubset(variants):
        raise ValueError("Requested variants must be unique and include all supplied signal variants")
    benchmark_map = benchmark.set_index("entry")["return"].to_dict()
    by_week = {(entry, variant): block for (entry, variant), block in
               joined.groupby(["entry", "variant"], sort=False)}
    empty = joined.iloc[:0]
    ic_rows, portfolio_rows, holding_rows, holding_chunks, excluded_rows = [], [], [], [], []
    previous: dict[tuple[str, str], tuple[dict, dict]] = {}
    previous_targets: dict[tuple[str, str], dict | None] = {}
    dead_portfolios: dict[tuple[str, str], str] = {}
    funded_portfolios: set[tuple[str, str]] = set()
    previous_exit = None

    for week in weeks.itertuples(index=False):
        if previous_exit is not None and week.entry != previous_exit:
            previous.clear()
            previous_targets.clear()
            funded_portfolios.clear()
        previous_exit = week.exit
        base_return = float(benchmark_map[week.entry])
        for variant in variants:
            block = by_week.get((week.entry, variant), empty)
            signal_known = len(block) > 0
            if not signal_known:
                excluded_rows.append({"entry": week.entry, "exit": week.exit, "variant": variant,
                                      "n_stocks": 0, "reason": "missing_signal"})
            masks = {"ALL": np.ones(len(block), dtype=bool), "TOP20": block.percentile > .8,
                     "TOP50": block.percentile > .5, "BOTTOM50": block.percentile <= .5,
                     "BOTTOM20": block.percentile <= .2}
            for subset, mask in masks.items():
                selected = block.loc[mask]
                for method in ("spearman", "pearson"):
                    ic_rows.append({"entry": week.entry, "exit": week.exit, "variant": variant,
                                    "method": method, "subset": subset, "n_stocks": len(selected),
                                    "signal_available": signal_known,
                                    "n_labels": int(selected.label_available.sum()),
                                    "label_coverage": float(selected.label_available.mean()) if len(selected) else np.nan,
                                    "ic": rank_ic(selected.signal, selected["return"], method)})
            period_results = {}
            for portfolio in PORTFOLIOS[:-1]:
                group = block if portfolio == "ALL" else block.loc[block.group == int(portfolio[1:])]
                key = (variant, portfolio)
                n = len(group)
                empty_cash = bool(empty_groups == "cash" and signal_known and n == 0)
                old, old_returns = previous.get(key, (None, None))
                drift_known = old is None or all(np.isfinite(old_returns[c]) for c in old)
                if old and drift_known and sum(old[c] * (1 + old_returns[c]) for c in old) <= 0:
                    dead_portfolios[key] = "bankruptcy"
                    previous.pop(key, None)
                dead = key in dead_portfolios
                target = dict.fromkeys(group.code, 1.0 / n) if n and not dead else {}
                initial_build = bool(n and not dead and key not in funded_portfolios)
                if initial_build:
                    funded_portfolios.add(key)
                turnover, drifted = np.nan, {}
                reason = f"portfolio_{dead_portfolios[key]}" if dead else "" if n else "empty_group"
                if not dead and drift_known and (signal_known or not available_case):
                    traded, drifted = drift_turnover(target, old, old_returns)
                    turnover = traded if n or empty_cash else np.nan
                old_target = previous_targets.get(key, {})
                target_turnover = (sum(abs(target.get(code, 0.) - old_target.get(code, 0.))
                                       for code in target.keys() | old_target.keys())
                                   if signal_known and not dead and old_target is not None else np.nan)
                previous_targets[key] = target if signal_known and not dead else None
                returns = group.set_index("code")["return"].to_dict()
                n_labels = int(group.label_available.sum())
                data_complete = bool(n_labels == n and (n or empty_cash))
                complete = bool(not dead and ((n and (n_labels > 0 if available_case else data_complete))
                                              or empty_cash))
                stock_return = (float(group["return"].mean()) if n and complete else
                                0. if empty_cash and complete else np.nan)
                if empty_cash and not dead:
                    reason = "empty_group_cash"
                elif not signal_known:
                    reason = "missing_signal"
                elif n and not dead and not complete:
                    reason = "missing_held_return"
                elif n and complete and not data_complete:
                    reason = "available_case_partial"
                if complete and n:
                    evaluation_weights = {code: 1. / n_labels if np.isfinite(value) else 0.
                                          for code, value in returns.items()}
                else:
                    evaluation_weights = {code: (0. if available_case else np.nan) for code in target}
                evaluation_weight_sum = (1. if n and complete else 0. if available_case or empty_cash else np.nan)
                cash_weight = 1. if empty_cash and not dead else 0. if n and not dead else np.nan
                missing_weight = 1. - n_labels / n if n else 0. if empty_cash else np.nan
                return_method = ("cash" if empty_cash and complete else
                                 "available_case" if available_case and complete else
                                 "frozen_weight" if complete else "unavailable")
                if stock_return == -1:
                    dead_portfolios[key] = "bankruptcy"
                    reason = "bankruptcy"
                restriction = float(group.trade_restricted.astype("boolean").mean()) if n and group.trade_restricted.notna().any() else np.nan
                row = {"entry": week.entry, "exit": week.exit, "variant": variant,
                       "portfolio": portfolio, "n_stocks": n, "return": stock_return,
                       "signal_available": signal_known,
                       "n_labels": n_labels,
                       "label_coverage": float(group.label_available.mean()) if n else np.nan,
                       "benchmark_return": base_return,
                       "excess_return": stock_return - base_return if complete else np.nan,
                       "turnover": turnover, "trade_restricted_fraction": restriction,
                       "initial_build": initial_build,
                       "target_turnover": target_turnover,
                       "trade_restriction_unknown_fraction": float(group.trade_restriction_unknown.mean()) if n else np.nan,
                       "turnover_complete": bool(drift_known and (n or empty_cash) and not dead),
                       "complete": complete, "data_complete": data_complete,
                       "missing_weight": missing_weight, "evaluation_weight_sum": evaluation_weight_sum,
                       "cash_weight": cash_weight, "return_method": return_method, "reason": reason}
                period_results[portfolio] = row
                portfolio_rows.append(row)
                ledger_codes = target.keys() | drifted.keys() | (old or {}).keys()
                if not signal_known:
                    ledger_codes = set()
                for code in sorted(ledger_codes):
                    holding_rows.append({"entry": week.entry, "exit": week.exit,
                                         "variant": variant, "portfolio": portfolio, "code": code,
                                         "target_weight": target.get(code, 0.0),
                                         "evaluation_weight": evaluation_weights.get(code, 0.0),
                                         "drifted_weight": drifted.get(code, 0.0) if drift_known else np.nan,
                                         "trade_weight": target.get(code, 0.0) - drifted.get(code, 0.0) if drift_known else np.nan,
                                         "return": returns.get(code, np.nan),
                                         "is_exit": code not in target})
                if (n or empty_cash) and key not in dead_portfolios:
                    previous[key] = (target, returns)
                else:
                    # Restart a new segment after a missing group; do not invent
                    # a cash portfolio or connect drift across the missing week.
                    previous.pop(key, None)
                    if not signal_known:
                        # An absent factor period is excluded, not invested in
                        # cash. The next observed period funds a new theoretical
                        # segment instead of drifting positions across the gap.
                        previous_targets.pop(key, None)
                        funded_portfolios.discard(key)
            high, low = period_results["Q10"], period_results["Q1"]
            spread_key = (variant, "Q10-Q1")
            complete = bool(high["complete"] and low["complete"] and spread_key not in dead_portfolios)
            spread_return = high["return"] - low["return"] if complete else np.nan
            spread_reason = (f"portfolio_{dead_portfolios[spread_key]}" if spread_key in dead_portfolios
                             else "" if complete else "missing_held_return" if
                             "missing_held_return" in (high["reason"], low["reason"]) else "empty_leg")
            if complete and spread_return <= -1:
                spread_reason = "bankruptcy" if spread_return == -1 else "insolvency"
                dead_portfolios[spread_key] = spread_reason

            def leg_diagnostic(column):
                # Empty legs contribute no stocks, not NaN times zero. Cash
                # on both legs has no restricted stock exposure.
                active = [leg for leg in (high, low) if leg["n_stocks"]]
                if not active:
                    return 0. if complete else np.nan
                if any(not np.isfinite(leg[column]) for leg in active):
                    return np.nan
                return sum(leg[column] * leg["n_stocks"] for leg in active) / sum(leg["n_stocks"] for leg in active)

            portfolio_rows.append({
                "entry": week.entry, "exit": week.exit, "variant": variant, "portfolio": "Q10-Q1",
                "signal_available": signal_known,
                "n_stocks": high["n_stocks"] + low["n_stocks"],
                "n_labels": high["n_labels"] + low["n_labels"],
                "label_coverage": ((high["n_labels"] + low["n_labels"]) / (high["n_stocks"] + low["n_stocks"]))
                                  if high["n_stocks"] + low["n_stocks"] else np.nan,
                "return": spread_return,
                "benchmark_return": base_return, "excess_return": np.nan,
                "turnover": high["turnover"] + low["turnover"] if complete or available_case else np.nan,
                "initial_build": bool(high["initial_build"] or low["initial_build"]),
                "target_turnover": high["target_turnover"] + low["target_turnover"],
                "trade_restricted_fraction": leg_diagnostic("trade_restricted_fraction") if complete else np.nan,
                "trade_restriction_unknown_fraction": leg_diagnostic("trade_restriction_unknown_fraction") if complete else np.nan,
                "turnover_complete": bool(high["turnover_complete"] and low["turnover_complete"]),
                "complete": complete, "data_complete": bool(high["data_complete"] and low["data_complete"]),
                "missing_weight": (high["missing_weight"] + low["missing_weight"]) / 2,
                "evaluation_weight_sum": high["evaluation_weight_sum"] + low["evaluation_weight_sum"] if complete else np.nan,
                "cash_weight": (high["cash_weight"] + low["cash_weight"]) / 2 if complete else np.nan,
                "return_method": ("available_case_spread" if available_case else "frozen_weight_spread") if complete else "unavailable",
                "reason": spread_reason})
            # Materialize one week's ledger at a time.  Keeping millions of
            # Python row dictionaries until the end unnecessarily multiplies
            # memory consumption for a complete A-share history.
            if holding_rows:
                holding_chunks.append(pd.DataFrame(holding_rows))
                holding_rows.clear()

    ic_weekly = pd.DataFrame(ic_rows).sort_values(["variant", "method", "subset", "entry"])
    ic_weekly["cumulative_ic"] = np.nan
    ic_weekly["segment_id"] = np.nan
    summary_rows = []
    for (variant, method, subset), block in ic_weekly.groupby(["variant", "method", "subset"], sort=False):
        block = block[block.signal_available]
        values = block.ic.to_numpy(dtype=float)
        cumulative, segments = _segmented_path(values, compound=False, entries=block.entry, exits=block.exit)
        ic_weekly.loc[block.index, "cumulative_ic"] = cumulative
        ic_weekly.loc[block.index, "segment_id"] = [np.nan if v is None else v for v in segments]
        good = values[np.isfinite(values)]
        n = len(good)
        mean = float(good.mean()) if n else np.nan
        sd = float(good.std(ddof=1)) if n > 1 else np.nan
        nw = mean_interval(_values_with_gaps(values, _calendar_breaks(len(block), block.entry, block.exit)), nw_lag)
        summary_rows.append({
            "variant": variant, "method": method, "subset": subset, "n": n,
            "signal_observations": int(block.n_stocks.sum()),
            "label_observations": int(block.n_labels.sum()),
            "label_coverage": float(block.n_labels.sum() / block.n_stocks.sum()) if block.n_stocks.sum() else np.nan,
            "ic_mean": mean, "ic_std": sd,
            "annual_icir": mean / sd * np.sqrt(annualization) if sd > 0 else np.nan,
            "t_stat": mean / sd * np.sqrt(n) if sd > 0 else np.nan,
            "win_rate": float(np.mean(good > 0)) if n else np.nan,
            "nw_t": nw["mean"] / nw["se"] if nw.get("se") and nw["se"] > 0 else np.nan,
            "nw_lower": nw.get("lower"), "nw_upper": nw.get("upper"), "nw_status": nw["status"]})

    portfolio_weekly = pd.DataFrame(portfolio_rows).sort_values(["variant", "portfolio", "entry"])
    for column in ("nav", "excess_nav", "segment_id", "excess_segment_id"):
        portfolio_weekly[column] = np.nan
    portfolio_weekly["turnover_one_way"] = portfolio_weekly.turnover / 2.
    portfolio_weekly["target_turnover_one_way"] = portfolio_weekly.target_turnover / 2.
    continuing = ~portfolio_weekly.initial_build & np.isfinite(portfolio_weekly.turnover_one_way)
    portfolio_weekly["high_turnover"] = pd.Series(pd.NA, index=portfolio_weekly.index, dtype="boolean")
    portfolio_weekly.loc[continuing, "high_turnover"] = (
        portfolio_weekly.loc[continuing, "turnover_one_way"] > turnover_warning_threshold)
    portfolio_weekly["turnover_warning_threshold"] = turnover_warning_threshold
    performance_rows, annual_rows = [], []
    for (variant, portfolio), block in portfolio_weekly.groupby(["variant", "portfolio"], sort=False):
        block = block[block.signal_available]
        nav, segments = _segmented_path(block["return"], entries=block.entry, exits=block.exit)
        excess_nav, excess_segments = _segmented_path(block.excess_return, entries=block.entry, exits=block.exit)
        portfolio_weekly.loc[block.index, "nav"] = nav
        portfolio_weekly.loc[block.index, "excess_nav"] = excess_nav
        portfolio_weekly.loc[block.index, "segment_id"] = [np.nan if v is None else v for v in segments]
        portfolio_weekly.loc[block.index, "excess_segment_id"] = [np.nan if v is None else v for v in excess_segments]
        kwargs = {"annualization": annualization, "turnover_warning_threshold": turnover_warning_threshold}
        excess = None if portfolio == "Q10-Q1" else block.excess_return
        metrics = performance_metrics(block["return"], block.turnover, excess,
                                      target_turnover=block.target_turnover, initial_build=block.initial_build,
                                      entries=block.entry, exits=block.exit, **kwargs)
        if block.empty:
            metrics["status"] = "no_data"
        if portfolio == "Q10-Q1":
            # (high - benchmark) - (low - benchmark) = high - low.
            # Keep legacy excess CAGR/nav absent for a self-financing spread.
            metrics["excess_sharpe"] = metrics["sharpe_ratio"]
            metrics["excess_sharpe_n_periods"] = metrics["sharpe_n_periods"]
        performance_rows.append({"variant": variant, "portfolio": portfolio, **metrics})
        for year, annual in block.groupby(block.entry.dt.year, sort=True):
            annual_excess = None if portfolio == "Q10-Q1" else annual.excess_return
            metrics = performance_metrics(annual["return"], annual.turnover, annual_excess,
                                          target_turnover=annual.target_turnover,
                                          initial_build=annual.initial_build,
                                          entries=annual.entry, exits=annual.exit, **kwargs)
            if portfolio == "Q10-Q1":
                metrics["excess_sharpe"] = metrics["sharpe_ratio"]
                metrics["excess_sharpe_n_periods"] = metrics["sharpe_n_periods"]
            annual_rows.append({"variant": variant, "portfolio": portfolio, "year": int(year), **metrics})

    holdings = (pd.concat(holding_chunks, ignore_index=True) if holding_chunks else
                pd.DataFrame(columns=["entry", "exit", "variant", "portfolio", "code",
                                      "target_weight", "evaluation_weight", "drifted_weight", "trade_weight", "return", "is_exit"]))
    # Keep signal provenance in the position ledger; exit rows have no current
    # signal and retain their explicit target weight of zero.
    metadata = [col for col in joined if col not in {"return", "unresolved", "trade_restricted"}]
    holdings = holdings.merge(joined[metadata], on=["entry", "exit", "variant", "code"],
                              how="left", validate="many_to_one")
    ic_weekly = ic_weekly.loc[ic_weekly.signal_available].drop(columns="signal_available")
    portfolio_weekly = portfolio_weekly.loc[portfolio_weekly.signal_available].drop(columns="signal_available")
    return {"ic_weekly": ic_weekly.reset_index(drop=True),
            "ic_summary": pd.DataFrame(summary_rows),
            "portfolio_weekly": portfolio_weekly.reset_index(drop=True),
            "performance": pd.DataFrame(performance_rows),
            "holdings": holdings,
            "annual_performance": pd.DataFrame(annual_rows),
            "excluded_periods": pd.DataFrame(excluded_rows, columns=["entry", "exit", "variant", "n_stocks", "reason"])}
