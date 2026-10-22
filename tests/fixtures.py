"""Small deterministic, explicitly synthetic datasets for framework acceptance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fundamental_factor.config import RISK_FIELDS, load_config


def signal_fixture(weeks_count: int = 2, stocks: int = 100, seed: int = 19):
    """An unlabelled one-factor panel; future returns are deliberately separate."""
    rng = np.random.default_rng(seed)
    entries = pd.date_range("2015-01-05", periods=weeks_count + 1, freq="W-MON")
    weeks = pd.DataFrame({"entry": entries[:-1], "exit": entries[1:]})
    weeks["cutoff"] = weeks.entry - pd.Timedelta(nanoseconds=1)
    n = weeks_count * stocks
    frame = pd.DataFrame({"entry": np.repeat(entries[:-1], stocks),
                          "exit": np.repeat(entries[1:], stocks),
                          "code": np.tile([f"{i:06d}.SZ" for i in range(stocks)], weeks_count),
                          "value": rng.normal(size=n),
                          "industry": np.tile(np.resize(["银行", "工业", "消费", "科技"], stocks), weeks_count),
                          "float_mcap": np.exp(rng.normal(20, 1, n))})
    frame = frame.merge(weeks[["entry", "cutoff"]], on="entry", validate="many_to_one")
    for name in RISK_FIELDS:
        frame[name] = rng.normal(size=n)
    return frame, weeks


def disk_fixture(root: Path, weeks_count: int = 36, stocks_count: int = 100):
    """Create prices, snapshots, risks and historical metadata; no /data/mongo IO."""
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    data.mkdir(exist_ok=True)
    rng = np.random.default_rng(88)
    first = pd.Timestamp("2015-01-05")
    entries = pd.date_range(first, periods=weeks_count + 1, freq="W-MON")
    trading = pd.bdate_range(first - pd.Timedelta(days=3), entries[-1])
    codes = [f"{i:06d}.SZ" for i in range(stocks_count)]
    # Friday snapshots supply Monday signals; Monday snapshots cannot enter that Monday.
    snapshot_index = np.searchsorted(entries, trading, side="right")
    roic = rng.normal(size=(weeks_count + 2, stocks_count))
    fund = rng.normal(size=(weeks_count + 2, stocks_count))
    fund[rng.random(fund.shape) < .35] = 0.
    levels = np.ones((weeks_count + 1, stocks_count)) * 100
    for k in range(weeks_count):
        q = pd.Series(roic[k]).rank(method="average", pct=True).to_numpy()
        f = pd.Series(fund[k]).rank(method="average", pct=True).to_numpy()
        returns = .002 + .008 * (q - .5) + .006 * (f - .5) + rng.normal(0, .008, stocks_count)
        levels[k + 1] = levels[k] * (1 + returns)
    quote_index = np.maximum(snapshot_index - 1, 0)
    grid = pd.DataFrame({"date": np.repeat(trading, stocks_count),
                         "code": np.tile(codes, len(trading))})
    for source, field, values in [("roic", "ROIC_new", roic), ("fund", "fundnum_promotion", fund)]:
        snapshots = grid.copy()
        snapshots[field] = values[snapshot_index].ravel()
        snapshots.to_feather(data / f"{source}.feather")
    quotes = grid.copy()
    quotes["acc_open"] = levels[quote_index].ravel()
    quotes["acc_close"] = quotes.acc_open
    quotes.to_feather(data / "quotes.feather")
    risk = grid.copy()
    for field in RISK_FIELDS:
        exposure = rng.normal(size=(weeks_count + 2, stocks_count))
        risk[field] = exposure[snapshot_index].ravel()
    risk.to_feather(data / "risk.feather")
    pd.DataFrame({"code": codes, "listing_date": pd.Timestamp("2010-01-01"),
                  "delisting_date": pd.NaT, "is_a_share": True}).to_parquet(data / "stocks.parquet", index=False)
    history = pd.DataFrame({"date": trading[0], "code": codes,
                            "industry": np.resize(["工业", "消费", "科技"], stocks_count),
                            "float_mcap": np.exp(rng.normal(20, 1, stocks_count)), "is_st": False})
    history.loc[0, "industry"] = "银行"
    history.loc[1, "industry"] = "非银金融"
    history.to_parquet(data / "history.parquet", index=False)
    status = grid.copy()
    status["suspended"], status["limit_up"], status["limit_down"] = False, False, False
    status.to_parquet(data / "status.parquet", index=False)
    pd.DataFrame({"date": pd.bdate_range(first - pd.Timedelta(days=500), entries[-1])}).to_csv(
        data / "calendar.csv", index=False)
    raw = {"schema_version": 2, "data_root": str(data), "output_dir": str(root / "run"),
           "data_policy": {"mode": "strict"},
           "classification_workbook": None, "dataset_kind": "synthetic",
           "factors": [{"name": "ROIC_new", "label": "ROIC", "source": "roic", "column": "ROIC_new"},
                       {"name": "fundnum_promotion", "label": "公募持仓变化", "source": "fund",
                        "column": "fundnum_promotion"}],
           "sources": {"roic": "roic.feather", "fund": "fund.feather", "quotes": "quotes.feather",
                       "risk": "risk.feather", "stocks": "stocks.parquet", "history": "history.parquet",
                       "calendar": "calendar.csv", "status": "status.parquet", "terminal": None},
           "point_in_time_confirmed": True,
           "point_in_time_evidence": "SYNTHETIC ONLY: point-in-time daily snapshots.",
           "price_contract": {"verified": True, "total_return": True, "suspended_total_return": True,
                              "description": "SYNTHETIC ONLY: adjusted total-return levels; no corporate actions."},
           "backtest": {"min_listing_sessions": 120}}
    path = root / "config.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return load_config(path)


def enable_execution_fixture(config):
    """No corporate actions in this synthetic world, so raw and total-return prices coincide."""
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    data = Path(raw["data_root"])
    quotes = pd.read_feather(data / "quotes.feather").rename(columns={"acc_open": "open", "acc_close": "close"})
    quotes.to_parquet(data / "raw_quotes.parquet", index=False)
    pd.DataFrame(columns=["date", "code", "share_ratio", "cash_per_share", "notice_date"]).to_csv(
        data / "actions.csv", index=False)
    raw["sources"].update(execution_quotes="raw_quotes.parquet", execution_status="status.parquet",
                          corporate_actions="actions.csv", execution_terminal=None)
    raw["execution"] = {"enabled": True, "initial_capital": 1_000_000., "raw_prices_verified": True,
                        "status_verified": True, "corporate_actions_complete": True,
                        "coverage_start": str(quotes.date.min().date()), "coverage_end": str(quotes.date.max().date()),
                        "description": "SYNTHETIC ONLY: no corporate actions; raw prices equal total-return prices."}
    config.path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return load_config(config.path)
