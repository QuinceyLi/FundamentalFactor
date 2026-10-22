"""Weekly open execution sensitivity with a persistent, cash-funded ledger.

This supplements the theoretical equal-weight factor portfolios. It does not
simulate borrowing, fees, taxes, board lots, daily order retries or short sales.
Raw prices use explicit distributions. The separate adjusted-unit mode uses
adjusted prices with no distribution or raw-share settlement events. Unknown
trade status freezes orders, never inventory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import performance_metrics
from .config import DataGateError


@dataclass
class _Account:
    variant: str
    portfolio: str
    cash: float
    shares: dict = field(default_factory=dict)
    marks: dict = field(default_factory=dict)
    mark_dates: dict = field(default_factory=dict)
    mark_sources: dict = field(default_factory=dict)
    pending: dict | None = None
    invalid_since: pd.Timestamp | None = None
    invalid_codes: set = field(default_factory=set)

    def nav(self):
        for code, number in self.shares.items():
            if number > 0 and (code not in self.marks or not np.isfinite(self.marks[code])
                               or self.marks[code] < 0):
                raise DataGateError(f"持仓 {code} 没有可用历史估值")
        return self.cash + sum(number * self.marks[code] for code, number in self.shares.items())


def _frame(frame, required, name, date_columns=("date",)):
    if frame is None:
        return pd.DataFrame(columns=required)
    result = frame.copy()
    missing = set(required) - set(result)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    for column in date_columns:
        if column in result:
            result[column] = pd.to_datetime(result[column]).dt.normalize()
    if "code" in result:
        if result.code.isna().any():
            raise ValueError(f"{name} has missing stock codes")
        result["code"] = result.code.astype(str)
    return result


def _unique(frame, keys, name):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name} needs nonmissing, unique {keys}")


def _positive(value):
    return value is not None and pd.notna(value) and np.isfinite(value) and value > 0


def _flag(value):
    return None if value is None or pd.isna(value) else bool(value)


def _mark(account, date, day_quote, day_status, field_name):
    carried = 0
    for code in account.shares:
        quote = day_quote.get(code, {})
        suspended = _flag(day_status.get(code, {}).get("suspended")) is True
        value = quote.get(field_name)
        if _positive(value) and not suspended:
            account.marks[code] = float(value)
            account.mark_dates[code] = date
            account.mark_sources[code] = field_name
        else:
            carried += 1
            if code not in account.marks or not np.isfinite(account.marks[code]):
                raise DataGateError(f"{date:%Y-%m-%d} {code} 缺少 {field_name} 及历史估值")
            account.mark_sources[code] = ("carry_suspended" if suspended else f"carry_missing_{field_name}")
    return carried


def _events(account, day_actions, day_terminal):
    dividend_cash, bonus_shares, terminal_cash = 0., 0., 0.
    for code, event in day_actions.items():
        number = account.shares.get(code, 0.)
        if number <= 0:
            continue
        if code not in account.marks:
            raise DataGateError(f"{code} 公司行动前缺少持仓估值")
        cash, ratio = float(event["cash_per_share"]), float(event["share_ratio"])
        adjusted = (account.marks[code] - cash) / (1 + ratio)
        if adjusted < -1e-10:
            raise DataGateError(f"{code} 分红大于前次估值，需要核对公司行动及价格")
        account.marks[code] = max(0., adjusted)
        account.mark_sources[code] = "corporate_action_adjusted"
        dividend_cash += number * cash
        bonus_shares += number * ratio
        account.shares[code] = number * (1 + ratio)
    account.cash += dividend_cash
    for code, event in day_terminal.items():
        number = account.shares.pop(code, 0.)
        terminal_cash += number * float(event["cash_per_share"])
    account.cash += terminal_cash
    return dividend_cash, bonus_shares, terminal_cash


def _block_reason(code, side, day_quote, day_status, terminated):
    if code in terminated:
        return terminated[code]
    state = day_status.get(code, {})
    suspended = _flag(state.get("suspended"))
    if suspended is True:
        return "suspended"
    if suspended is None:
        return "unknown_status"
    limit = _flag(state.get("limit_up" if side == "buy" else "limit_down"))
    if limit is None:
        return "unknown_limit_status"
    if limit:
        return "limit_up" if side == "buy" else "limit_down"
    if not _positive(day_quote.get(code, {}).get("open")):
        return "missing_open"
    return ""


def _rebalance(account, date, target_codes, day_quote, day_status, terminated):
    nav = account.nav()
    target = {code: 1 / len(target_codes) for code in target_codes} if target_codes else {}
    codes = sorted(set(target) | set(account.shares))
    orders, rows = [], []
    for code in codes:
        before = account.shares.get(code, 0.)
        held_value = before * account.marks.get(code, 0.)
        requested = nav * target.get(code, 0.) - held_value
        if abs(requested) <= max(1e-9, nav * 1e-12):
            continue
        side = "buy" if requested > 0 else "sell"
        raw_open = day_quote.get(code, {}).get("open")
        price = float(raw_open) if _positive(raw_open) else np.nan
        reason = _block_reason(code, side, day_quote, day_status, terminated)
        requested_shares = requested / price if np.isfinite(price) else np.nan
        # A full exit is based on existing shares, never on a stale mark ratio.
        if side == "sell" and target.get(code, 0.) == 0:
            requested_shares = -before
        if side == "sell" and np.isfinite(requested_shares):
            requested_shares = max(-before, requested_shares)
        orders.append(dict(code=code, side=side, price=price, requested_value=requested,
                           requested_shares=requested_shares, rejection_reason=reason,
                           executed_shares=0., executed_value=0., cash_scaled=False))
    for order in orders:
        if order["side"] != "sell" or order["rejection_reason"]:
            continue
        code, quantity = order["code"], order["requested_shares"]
        amount = quantity * order["price"]
        account.shares[code] += quantity
        account.cash -= amount
        order.update(executed_shares=quantity, executed_value=amount)
        if account.shares[code] <= 1e-10:
            account.shares.pop(code)
    purchases = [order for order in orders if order["side"] == "buy" and not order["rejection_reason"]]
    buy_total = sum(order["requested_value"] for order in purchases)
    scale = min(1., max(0., account.cash) / buy_total) if buy_total > 0 else 1.
    for order in purchases:
        code = order["code"]
        amount = order["requested_value"] * scale
        quantity = amount / order["price"]
        account.cash -= amount
        if quantity > 0:
            account.shares[code] = account.shares.get(code, 0.) + quantity
            account.marks[code] = order["price"]
            account.mark_dates[code] = date
            account.mark_sources[code] = "open"
        order.update(executed_shares=quantity, executed_value=amount,
                     cash_scaled=scale < 1 - 1e-12,
                     rejection_reason="cash_shortfall" if scale < 1 - 1e-12 else "")
    if account.cash < -max(1e-8, nav * 1e-12):
        raise DataGateError("执行账本产生负现金")
    account.cash = max(0., account.cash)
    if not np.isclose(account.nav(), nav, atol=1e-7, rtol=1e-10):
        raise DataGateError("费前开盘调仓未保持资产守恒")
    for code in codes:
        number = account.shares.get(code, 0.)
        mark = account.marks.get(code, np.nan)
        rows.append(dict(code=code, target_weight=target.get(code, 0.),
                         actual_weight=number * mark / nav if number and nav > 0 else 0.,
                         shares=number, mark=mark, mark_date=account.mark_dates.get(code, pd.NaT),
                         mark_source=account.mark_sources.get(code, "unvalued_unheld")))
    traded = sum(abs(order["executed_value"]) for order in orders)
    blocked = {side: sum(abs(order["requested_value"] - order["executed_value"])
                         for order in orders if order["side"] == side) for side in ("buy", "sell")}
    return dict(nav_start=nav, turnover=traded / nav if nav > 0 else np.nan,
                cash_weight=account.cash / nav if nav > 0 else np.nan,
                n_stocks=len(account.shares), blocked_buy_value=blocked["buy"],
                blocked_sell_value=blocked["sell"], target_empty=not bool(target)), rows, orders


def run_execution(signals, quotes, status, actions, weeks, calendar, terminal=None,
                  initial_capital=1_000_000., portfolios=None, annualization=52,
                  price_mode="raw", delistings=None):
    """Return weekly, holdings, trades, daily and performance DataFrames.

    Each variant has independent ALL/Q1..Q10 long-only accounts. Weekly return
    is entry pre-trade NAV to exit pre-trade NAV; cash earns zero. End-point
    distributions belong to the old inventory. Quotes never fill backward.
    Terminal ``date`` is the cash settlement date and ``available_date`` cannot
    be later. Event source completeness must be audited by the caller.
    ``carried_valuation_count`` counts stock/valuation-time carry observations.
    Daily rows are close NAV except the final exit (``stage=open_final``).
    Adjusted units are synthetic quantities, not real shares. An account holding
    an unresolved delisting reports missing returns from that holding period and
    stops trading. Its carried values remain available only as indicative marks.
    """
    if price_mode not in ("raw", "adjusted_units"):
        raise ValueError("price_mode must be raw or adjusted_units")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be positive and finite")
    if not isinstance(annualization, (int, np.integer)) or annualization <= 0:
        raise ValueError("annualization must be a positive integer")
    signals = _frame(signals, ["entry", "exit", "code", "variant", "group"], "signals", ("entry", "exit"))
    quotes = _frame(quotes, ["date", "code", "open", "close"], "quotes")
    status = _frame(status, ["date", "code", "suspended", "limit_up", "limit_down"], "status")
    actions = _frame(actions, ["date", "code", "share_ratio", "cash_per_share", "notice_date"], "actions", ("date", "notice_date"))
    terminal = _frame(terminal, ["date", "code", "cash_per_share", "available_date"], "terminal", ("date", "available_date"))
    if price_mode == "adjusted_units" and (not actions.empty or not terminal.empty):
        raise ValueError("Adjusted-unit prices cannot be combined with raw-share corporate actions or terminal cash")
    if price_mode == "raw" and delistings is not None:
        raise ValueError("Raw-share delisting resolution remains the caller's validated cash-settlement contract")
    delistings = _frame(delistings, ["code", "delisting_date"], "delistings", ("delisting_date",))
    delistings = delistings[delistings.delisting_date.notna()].copy()
    _unique(delistings, ["code"], "delistings")
    weeks = _frame(weeks, ["entry", "exit"], "weeks", ("entry", "exit")).sort_values("entry")
    _unique(signals, ["entry", "variant", "code"], "signals")
    _unique(weeks, ["entry"], "weeks")
    if signals.empty or weeks.empty:
        raise DataGateError("执行账本需要非空信号及完整持有周次")
    if (weeks.exit <= weeks.entry).any() or weeks.exit.duplicated().any():
        raise ValueError("weeks must have increasing, unique entry/exit dates")
    if (weeks.entry.iloc[1:].to_numpy() < weeks.exit.iloc[:-1].to_numpy()).any():
        raise ValueError("execution weeks cannot overlap")
    pairs = weeks.set_index("entry").exit
    if not signals.entry.isin(pairs.index).all() or not (signals.exit.to_numpy() == signals.entry.map(pairs).to_numpy()).all():
        raise ValueError("signal holding periods do not match weeks")
    if signals.group.notna().any() and not signals.loc[signals.group.notna(), "group"].isin(range(1, 11)).all():
        raise ValueError("signal groups must be Q1..Q10 or missing")
    days = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().sort_values().unique()
    if days.hasnans or not weeks.entry.isin(days).all() or not weeks.exit.isin(days).all():
        raise DataGateError("周度入场/结束日缺少交易日历")
    start, end = weeks.entry.min(), weeks.exit.max()
    days = days[(days >= start) & (days <= end)]
    for name, frame in (("quotes", quotes), ("status", status), ("actions", actions), ("terminal", terminal)):
        _unique(frame, ["date", "code"], name)
        active = frame.date.between(start, end)
        if not frame.loc[active, "date"].isin(days).all():
            raise DataGateError(f"{name} 存在交易日历之外的日期")
    for frame, value_cols, notice in ((actions, ["share_ratio", "cash_per_share"], "notice_date"),
                                      (terminal, ["cash_per_share"], "available_date")):
        if not np.isfinite(frame[value_cols].to_numpy(dtype=float)).all() or (frame[value_cols] < 0).any().any():
            raise ValueError("Corporate actions / terminal values must be finite and nonnegative")
        if frame[notice].isna().any() or (frame[notice] > frame.date).any():
            raise DataGateError(f"{notice} 晚于事件日期或缺失，不能使用未来信息记账")
    for column in ("suspended", "limit_up", "limit_down"):
        if not status[column].dropna().isin([True, False, 0, 1]).all():
            raise ValueError(f"status.{column} must contain nullable booleans")
    quote_days = {date: day.set_index("code") for date, day in quotes.groupby("date", sort=False)}
    status_days = {date: day.set_index("code") for date, day in status.groupby("date", sort=False)}
    action_days = {date: day.set_index("code") for date, day in actions.groupby("date", sort=False)}
    terminal_days = {date: day.set_index("code") for date, day in terminal.groupby("date", sort=False)}
    names = ["ALL"] + [f"Q{i}" for i in range(1, 11)] if portfolios is None else list(portfolios)
    if not names or len(set(names)) != len(names) or any(name not in ["ALL"] + [f"Q{i}" for i in range(1, 11)] for name in names):
        raise ValueError("portfolios must be unique names from ALL, Q1..Q10")
    variants = sorted(signals.variant.unique())
    accounts = [_Account(variant, name, float(initial_capital)) for variant in variants for name in names]
    targets = {}
    valid_signals = set()
    for (entry, variant), frame in signals.groupby(["entry", "variant"], sort=False):
        if frame.group.notna().any():
            valid_signals.add((entry, variant))
        for name in names:
            targets[(entry, variant, name)] = set(frame.code if name == "ALL" else frame.loc[frame.group.eq(int(name[1:])), "code"])
    records = {key: [] for key in ("weekly", "holdings", "trades", "daily", "performance")}
    terminated = {}
    for date in days:
        def day_dict(source):
            return source[date].to_dict("index") if date in source else {}
        day_quote, day_status = day_dict(quote_days), day_dict(status_days)
        day_actions, day_terminal = day_dict(action_days), day_dict(terminal_days)
        terminated.update({code: "terminal_settled" for code in day_terminal})
        if price_mode == "adjusted_units":
            terminated.update({code: "delisted_unresolved" for code in
                               delistings.loc[delistings.delisting_date.le(date), "code"]})
        for account in accounts:
            identity = dict(variant=account.variant, portfolio=account.portfolio)
            if price_mode == "adjusted_units":
                unresolved_held = {code for code, number in account.shares.items()
                                   if number > 0 and terminated.get(code) == "delisted_unresolved"}
                if unresolved_held:
                    account.invalid_since = account.invalid_since or date
                    account.invalid_codes.update(unresolved_held)
            invalid = account.invalid_since is not None
            quality = dict(price_mode=price_mode,
                           quality_status="unresolved_delisting" if invalid else "ok",
                           quality_reason="unresolved_delisting:" + ",".join(sorted(account.invalid_codes)) if invalid else "")
            dividend, bonus, redemption = _events(account, day_actions, day_terminal)
            carried = _mark(account, date, day_quote, day_status, "open")
            nav_open = account.nav()
            if account.pending is not None:
                account.pending["carried_valuation_count"] += carried
                if date == account.pending["exit"]:
                    row = account.pending
                    row.update(nav=np.nan if invalid else nav_open, indicative_nav=nav_open, **quality,
                               **{"return": nav_open / row["nav_start"] - 1 if not invalid and row["nav_start"] > 0 else np.nan})
                    records["weekly"].append(row)
                    account.pending = None
            if date in pairs.index:
                signal_missing = (date, account.variant) not in valid_signals
                if signal_missing or invalid:
                    # Failed/absent preprocessing must not manufacture a cash
                    # timing strategy. Preserve quantities and existing cash.
                    metadata = dict(nav_start=np.nan if invalid else nav_open, turnover=0. if nav_open > 0 and not invalid else np.nan,
                        cash_weight=account.cash / nav_open if nav_open > 0 and not invalid else np.nan,
                        n_stocks=len(account.shares), blocked_buy_value=0., blocked_sell_value=0., target_empty=False)
                    holdings = [dict(code=code, target_weight=np.nan,
                        actual_weight=number * account.marks[code] / nav_open if nav_open > 0 and not invalid else np.nan,
                        shares=number, mark=account.marks[code], mark_date=account.mark_dates[code],
                        mark_source=account.mark_sources[code]) for code, number in sorted(account.shares.items())]
                    trades = []
                else:
                    metadata, holdings, trades = _rebalance(account, date, targets.get((date, account.variant, account.portfolio), set()),
                                                           day_quote, day_status, terminated)
                period = dict(entry=date, exit=pairs.loc[date], **identity)
                account.pending = dict(**period, **metadata, indicative_nav_start=nav_open,
                                       carried_valuation_count=0, signal_missing=signal_missing)
                records["holdings"].extend({**period, **row} for row in holdings)
                records["trades"].extend({**period, **row} for row in trades)
            close_carried = 0 if date == end else _mark(account, date, day_quote, day_status, "close")
            if account.pending is not None:
                account.pending["carried_valuation_count"] += close_carried
            records["daily"].append(dict(date=date, **identity, **quality,
                nav=np.nan if invalid else account.nav(), indicative_nav=account.nav(), cash=account.cash,
                n_stocks=len(account.shares), carried_valuation_count=carried + close_carried,
                dividend_cash=dividend, bonus_shares=bonus, terminal_cash=redemption,
                stage="open_final" if date == end else "close"))
    weekly = pd.DataFrame(records["weekly"]).sort_values(["variant", "portfolio", "entry"]).reset_index(drop=True)
    for (variant, portfolio), rows in weekly.groupby(["variant", "portfolio"], sort=False):
        records["performance"].append(dict(variant=variant, portfolio=portfolio,
                                           **performance_metrics(rows["return"], rows.turnover, annualization=annualization)))
    trade_columns = ["entry", "exit", "variant", "portfolio", "code", "side", "price", "requested_value", "requested_shares",
                     "rejection_reason", "executed_shares", "executed_value", "cash_scaled"]
    holding_columns = ["entry", "exit", "variant", "portfolio", "code", "target_weight", "actual_weight", "shares", "mark", "mark_date", "mark_source"]
    result = {"weekly": weekly, "holdings": pd.DataFrame(records["holdings"], columns=holding_columns),
            "trades": pd.DataFrame(records["trades"], columns=trade_columns),
            "daily": pd.DataFrame(records["daily"]), "performance": pd.DataFrame(records["performance"])}
    if price_mode == "adjusted_units":
        result["holdings"] = result["holdings"].rename(columns={"shares": "adjusted_units"})
        result["trades"] = result["trades"].rename(columns={"requested_shares": "requested_units",
                                                          "executed_shares": "executed_units"})
        result["daily"] = result["daily"].rename(columns={"bonus_shares": "bonus_units"})
    return result
