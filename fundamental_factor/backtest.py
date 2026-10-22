"""Weekly, frozen-weight factor tests; no predictive model is fitted here.

Returns are gross statistical portfolio returns.  The position ledger is the
source of turnover: it records drifted old positions, targets, and exits instead
of approximating turnover from the fraction of changed group members.
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

    The caller preserves frozen signal membership and reports label coverage;
    incomplete portfolio returns are not inferred from this paired subset.
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
    result = {"n_periods": n, "total_return": np.nan, "annual_return": np.nan,
              "annual_volatility": volatility, "information_ratio": np.nan,
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


def performance_metrics(returns, turnover=None, excess_returns=None,
                        annualization: int = 52) -> dict:
    """Attachment-compatible CAGR/volatility statistics on a full weekly grid.

    Gaps are never compressed for cumulative results.  Valid-period volatility
    and win rate remain visible, but total/CAGR/drawdown and annual turnover are
    undefined whenever a full continuous investment history is unavailable.
    """
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    values = np.asarray(returns, dtype=float)
    result = _return_metrics(values, annualization)
    result["annual_turnover"] = np.nan
    if turnover is not None:
        traded = np.asarray(turnover, dtype=float)
        if traded.shape != values.shape:
            raise ValueError("Returns and turnover must share the same weekly grid")
        if result["status"] == "ok" and np.isfinite(traded).all():
            result["annual_turnover"] = float(traded.mean() * annualization)
    excess = (_return_metrics(excess_returns, annualization) if excess_returns is not None
              else {key: np.nan for key in _return_metrics([], annualization)})
    if excess_returns is not None and np.asarray(excess_returns).shape != values.shape:
        raise ValueError("Returns and excess returns must share the same weekly grid")
    for key in ("total_return", "annual_return", "annual_volatility", "information_ratio",
                "win_rate", "max_drawdown"):
        result[f"excess_{key}"] = excess[key]
    if excess_returns is not None and excess["status"] in {"bankruptcy", "insolvency"}:
        result["status"] += f";excess_{excess['status']}"
    return result


def _segmented_path(values, compound: bool = True) -> tuple[list[float], list[int | None]]:
    """Rebase valid runs after gaps; never connect an invalid gap with a line."""
    path, ids, running, segment, active, bankrupt = [], [], 1.0 if compound else 0.0, 0, False, False
    terminal_value = np.nan
    for value in np.asarray(values, dtype=float):
        if bankrupt:
            path.append(terminal_value)
            ids.append(segment if np.isfinite(terminal_value) else None)
            continue
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
                    weeks: pd.DataFrame, missing_returns: str = "error"
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if missing_returns not in ("error", "report"):
        raise ValueError("missing_returns must be error or report")
    allow_missing = missing_returns == "report"
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
    if not allow_missing and not np.isfinite(bench["return"].to_numpy(dtype=float)).all():
        raise DataGateError("The independent base-universe benchmark has unresolved weekly returns")
    if "unresolved" in bench:
        unresolved_benchmark = bench.unresolved.astype("boolean").fillna(True)
        if not allow_missing and unresolved_benchmark.any():
            raise DataGateError("The independent base-universe benchmark has unresolved labels")
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
    # The runner supports a single variant for pure tests, but any supplied
    # variants must contain exactly the same frozen stock sample each week.
    for _, block in signals.groupby(["entry", "exit"], sort=False):
        sets = [set(block.loc[block.variant == variant, "code"]) for variant in variants]
        if any(stock_set != sets[0] for stock_set in sets[1:]):
            raise DataGateError("Preprocessing variants do not share the same frozen stock sample")
    return joined, labels, bench, weeks


def evaluate_factor(signals: pd.DataFrame, labels: pd.DataFrame, benchmark: pd.DataFrame,
                    weeks: pd.DataFrame, annualization: int = 52,
                    nw_lag: int = 13, missing_returns: str = "error") -> dict[str, pd.DataFrame]:
    """Evaluate one factor and all preprocessing variants on identical weekly keys."""
    joined, labels, benchmark, weeks = _prepare_inputs(signals, labels, benchmark, weeks, missing_returns)
    variants = sorted(joined.variant.unique())
    benchmark_map = benchmark.set_index("entry")["return"].to_dict()
    by_week = {(entry, variant): block for (entry, variant), block in
               joined.groupby(["entry", "variant"], sort=False)}
    empty = joined.iloc[:0]
    ic_rows, portfolio_rows, holding_rows, holding_chunks = [], [], [], []
    previous: dict[tuple[str, str], tuple[dict, dict]] = {}
    dead_portfolios: dict[tuple[str, str], str] = {}

    for week in weeks.itertuples(index=False):
        base_return = float(benchmark_map[week.entry])
        for variant in variants:
            block = by_week.get((week.entry, variant), empty)
            masks = {"ALL": np.ones(len(block), dtype=bool), "TOP20": block.percentile > .8,
                     "TOP50": block.percentile > .5, "BOTTOM50": block.percentile <= .5,
                     "BOTTOM20": block.percentile <= .2}
            for subset, mask in masks.items():
                selected = block.loc[mask]
                for method in ("spearman", "pearson"):
                    ic_rows.append({"entry": week.entry, "exit": week.exit, "variant": variant,
                                    "method": method, "subset": subset, "n_stocks": len(selected),
                                    "n_labels": int(selected.label_available.sum()),
                                    "label_coverage": float(selected.label_available.mean()) if len(selected) else np.nan,
                                    "ic": rank_ic(selected.signal, selected["return"], method)})
            period_results = {}
            for portfolio in PORTFOLIOS[:-1]:
                group = block if portfolio == "ALL" else block.loc[block.group == int(portfolio[1:])]
                key = (variant, portfolio)
                n = len(group)
                old, old_returns = previous.get(key, (None, None))
                drift_known = old is None or all(np.isfinite(old_returns[c]) for c in old)
                if old is not None and drift_known and sum(old[c] * (1 + old_returns[c]) for c in old) <= 0:
                    dead_portfolios[key] = "bankruptcy"
                    previous.pop(key, None)
                dead = key in dead_portfolios
                target = dict.fromkeys(group.code, 1.0 / n) if n and not dead else {}
                turnover, drifted = np.nan, {}
                reason = f"portfolio_{dead_portfolios[key]}" if dead else "" if n else "empty_group"
                if not dead and drift_known:
                    traded, drifted = drift_turnover(target, old, old_returns)
                    turnover = traded if n else np.nan
                returns = group.set_index("code")["return"].to_dict()
                complete = bool(n and not dead and group.label_available.all())
                stock_return = float(group["return"].mean()) if complete else np.nan
                if n and not dead and not complete:
                    reason = "missing_held_return"
                if stock_return == -1:
                    dead_portfolios[key] = "bankruptcy"
                    reason = "bankruptcy"
                restriction = float(group.trade_restricted.astype("boolean").mean()) if n and group.trade_restricted.notna().any() else np.nan
                row = {"entry": week.entry, "exit": week.exit, "variant": variant,
                       "portfolio": portfolio, "n_stocks": n, "return": stock_return,
                       "n_labels": int(group.label_available.sum()),
                       "label_coverage": float(group.label_available.mean()) if n else np.nan,
                       "benchmark_return": base_return,
                       "excess_return": stock_return - base_return if n else np.nan,
                       "turnover": turnover, "trade_restricted_fraction": restriction,
                       "trade_restriction_unknown_fraction": float(group.trade_restriction_unknown.mean()) if n else np.nan,
                       "turnover_complete": bool(drift_known and n and not dead),
                       "complete": complete, "reason": reason}
                period_results[portfolio] = row
                portfolio_rows.append(row)
                for code in sorted(target.keys() | drifted.keys() | (old or {}).keys()):
                    holding_rows.append({"entry": week.entry, "exit": week.exit,
                                         "variant": variant, "portfolio": portfolio, "code": code,
                                         "target_weight": target.get(code, 0.0),
                                         "drifted_weight": drifted.get(code, 0.0) if drift_known else np.nan,
                                         "trade_weight": target.get(code, 0.0) - drifted.get(code, 0.0) if drift_known else np.nan,
                                         "return": returns.get(code, np.nan),
                                         "is_exit": code not in target})
                if n and key not in dead_portfolios:
                    previous[key] = (target, returns)
                else:
                    # Restart a new segment after a missing group; do not invent
                    # a cash portfolio or connect drift across the missing week.
                    previous.pop(key, None)
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
            portfolio_rows.append({
                "entry": week.entry, "exit": week.exit, "variant": variant, "portfolio": "Q10-Q1",
                "n_stocks": high["n_stocks"] + low["n_stocks"],
                "n_labels": high["n_labels"] + low["n_labels"],
                "label_coverage": ((high["n_labels"] + low["n_labels"]) / (high["n_stocks"] + low["n_stocks"]))
                                  if high["n_stocks"] + low["n_stocks"] else np.nan,
                "return": spread_return,
                "benchmark_return": base_return, "excess_return": np.nan,
                "turnover": high["turnover"] + low["turnover"] if complete else np.nan,
                "trade_restricted_fraction": (high["trade_restricted_fraction"] * high["n_stocks"] +
                                              low["trade_restricted_fraction"] * low["n_stocks"]) /
                                             (high["n_stocks"] + low["n_stocks"]) if complete else np.nan,
                "trade_restriction_unknown_fraction": (high["trade_restriction_unknown_fraction"] * high["n_stocks"] +
                                                        low["trade_restriction_unknown_fraction"] * low["n_stocks"]) /
                                                       (high["n_stocks"] + low["n_stocks"]) if complete else np.nan,
                "turnover_complete": bool(high["turnover_complete"] and low["turnover_complete"]),
                "complete": complete, "reason": spread_reason})
            # Materialize one week's ledger at a time.  Keeping millions of
            # Python row dictionaries until the end unnecessarily multiplies
            # memory consumption for a complete A-share history.
            if holding_rows:
                holding_chunks.append(pd.DataFrame(holding_rows))
                holding_rows.clear()

    ic_weekly = pd.DataFrame(ic_rows).sort_values(["variant", "method", "subset", "entry"])
    summary_rows = []
    for (variant, method, subset), block in ic_weekly.groupby(["variant", "method", "subset"], sort=False):
        values = block.ic.to_numpy(dtype=float)
        cumulative, segments = _segmented_path(values, compound=False)
        ic_weekly.loc[block.index, "cumulative_ic"] = cumulative
        ic_weekly.loc[block.index, "segment_id"] = [np.nan if v is None else v for v in segments]
        good = values[np.isfinite(values)]
        n = len(good)
        mean = float(good.mean()) if n else np.nan
        sd = float(good.std(ddof=1)) if n > 1 else np.nan
        nw = mean_interval(values, nw_lag)
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
    performance_rows, annual_rows = [], []
    for (variant, portfolio), block in portfolio_weekly.groupby(["variant", "portfolio"], sort=False):
        nav, segments = _segmented_path(block["return"])
        excess_nav, excess_segments = _segmented_path(block.excess_return)
        portfolio_weekly.loc[block.index, "nav"] = nav
        portfolio_weekly.loc[block.index, "excess_nav"] = excess_nav
        portfolio_weekly.loc[block.index, "segment_id"] = [np.nan if v is None else v for v in segments]
        portfolio_weekly.loc[block.index, "excess_segment_id"] = [np.nan if v is None else v for v in excess_segments]
        kwargs = {"annualization": annualization}
        excess = None if portfolio == "Q10-Q1" else block.excess_return
        metrics = performance_metrics(block["return"], block.turnover, excess, **kwargs)
        performance_rows.append({"variant": variant, "portfolio": portfolio, **metrics})
        for year, annual in block.groupby(block.entry.dt.year, sort=True):
            annual_excess = None if portfolio == "Q10-Q1" else annual.excess_return
            metrics = performance_metrics(annual["return"], annual.turnover, annual_excess, **kwargs)
            annual_rows.append({"variant": variant, "portfolio": portfolio, "year": int(year), **metrics})

    holdings = (pd.concat(holding_chunks, ignore_index=True) if holding_chunks else
                pd.DataFrame(columns=["entry", "exit", "variant", "portfolio", "code",
                                      "target_weight", "drifted_weight", "trade_weight", "return", "is_exit"]))
    # Keep signal provenance in the position ledger; exit rows have no current
    # signal and retain their explicit target weight of zero.
    metadata = [col for col in joined if col not in {"return", "unresolved", "trade_restricted"}]
    holdings = holdings.merge(joined[metadata], on=["entry", "exit", "variant", "code"],
                              how="left", validate="many_to_one")
    return {"ic_weekly": ic_weekly.reset_index(drop=True),
            "ic_summary": pd.DataFrame(summary_rows),
            "portfolio_weekly": portfolio_weekly.reset_index(drop=True),
            "performance": pd.DataFrame(performance_rows),
            "holdings": holdings,
            "annual_performance": pd.DataFrame(annual_rows)}
