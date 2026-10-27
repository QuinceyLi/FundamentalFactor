"""Batch orchestration of independent single-factor reports."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from .backtest import evaluate_factor, performance_metrics
from .config import Config, DataGateError
from .data import read_frame, write_json
from .panel import ensure_panel


def save_result_tables(directory, tables, holdings_csv_limit=1_000_000):
    """Large stock ledgers use lossless Parquet, avoiding multi-GB duplicate CSV."""
    artifacts = {}
    for name, table in tables.items():
        formats = []
        csv = directory / f"{name}.csv"
        if name != "holdings" or len(table) <= holdings_csv_limit:
            table.to_csv(csv, index=False)
            formats.append("csv")
        elif csv.exists():
            # A previous interrupted export must not masquerade as a full ledger.
            csv.unlink()
        if name == "holdings":
            table.to_parquet(directory / "holdings.parquet", index=False)
            formats.append("parquet")
        artifacts[name] = {"rows": len(table), "formats": formats}
        # Neutral names describe both weekly and monthly data. Historical names
        # remain available for existing notebooks and report readers.
        if name in ("ic_weekly", "portfolio_weekly"):
            alias = name.replace("_weekly", "_periodic")
            table.to_csv(directory / f"{alias}.csv", index=False)
            artifacts[alias] = {"rows": len(table), "formats": ["csv"]}
    return artifacts


def unresolved_held_delistings(holdings, universe, settled):
    """Only positions actually carried into delisting require an exit resolution."""
    held = holdings.merge(universe.drop_duplicates(["code", "delisting_date"]),
                          on="code", how="left", validate="many_to_one")
    affected = held[held.shares.gt(0) & held.delisting_date.notna() & held.delisting_date.le(held.exit)]
    return set(affected.code) - set(settled)


def _load_execution_inputs(config: Config, weeks: pd.DataFrame):
    """Validate the optional account independently of factor-statistic inputs."""
    execution_settings = config.raw.get("execution", {})
    execution_mode = execution_settings.get("price_mode", "raw")
    execution_inputs, universe, settled = None, None, set()
    if execution_settings.get("enabled"):
        from .audit import source_columns
        checked = json.loads((config.output / "audit.json").read_text())
        if not checked.get("execution_gate_passed", True):
            raise DataGateError("；".join(checked.get("execution_blocking", [])))
        if "coverage_start" in execution_settings or "coverage_end" in execution_settings:
            first = pd.Timestamp(execution_settings.get("coverage_start"))
            last = pd.Timestamp(execution_settings.get("coverage_end"))
            if pd.isna(first) or pd.isna(last) or first > weeks.entry.min() or last < weeks.exit.max():
                raise DataGateError("执行账本已核实覆盖期不足完整持有期；不可用缺失事件冒充无公司行动")
        required = source_columns(config)
        execution_inputs = {}
        if execution_mode == "adjusted_units":
            prices = read_frame(config.source("quotes"), required["quotes"])
            contract = config.raw["price_contract"]
            prices = prices.rename(columns={contract.get("open_column", "acc_open"): "open",
                                             contract.get("close_column", "acc_close"): "close"})
            execution_inputs["execution_quotes"] = prices[prices.date.between(weeks.entry.min(), weeks.exit.max())]
            del prices
            execution_inputs["corporate_actions"] = None
            execution_inputs["execution_terminal"] = None
            names = ("execution_status",)
        else:
            names = ("execution_quotes", "execution_status", "corporate_actions", "execution_terminal")
        for name in names:
            path = config.source(name)
            if path is None and name == "execution_terminal":
                execution_inputs[name] = None
            else:
                table = read_frame(path, required[name], nullable_booleans=name == "execution_status")
                execution_inputs[name] = table[table.date.between(weeks.entry.min(), weeks.exit.max())]
        execution_inputs["calendar"] = read_frame(config.source("calendar"), ["date"]).date
        # A theoretical terminal total-return level cannot be reused as raw cash per share.
        universe = pd.read_parquet(config.output / "universe.parquet", columns=["code", "delisting_date"])
        terminal = execution_inputs["execution_terminal"]
        settled = set()
        if terminal is not None:
            if terminal.code.duplicated().any():
                raise DataGateError("执行终止事件每代码只能提供一次最终现金结算；不支持未建模的分期结算")
            all_matches = universe.drop_duplicates(["code", "delisting_date"]).merge(
                terminal[["code", "date"]], on="code", how="inner")
            if (all_matches.delisting_date.isna() | all_matches.date.lt(all_matches.delisting_date)).any():
                raise DataGateError("执行终止结算日期早于退市日，或股票主表尚无对应退市日")
            settled = set(terminal.code)
    return execution_inputs, universe, settled


def run_research(config: Config) -> dict:
    manifest = ensure_panel(config)
    features = pd.read_parquet(config.output / "features.parquet")
    labels = pd.read_parquet(config.output / "labels.parquet")
    weeks = pd.read_parquet(config.output / "weeks.parquet")
    benchmark = pd.read_csv(config.output / "benchmark.csv", parse_dates=["entry", "exit"])
    coverage = pd.read_csv(config.output / "coverage.csv")
    execution_settings = config.raw.get("execution", {})
    execution_mode = execution_settings.get("price_mode", "raw")
    classic = config.data_policy["mode"] == "classic"
    execution_card = {"requested": bool(execution_settings.get("enabled")), "enabled": False,
                      "status": "disabled", "blocking": []}
    try:
        execution_inputs, universe, settled = _load_execution_inputs(config, weeks)
    except (DataGateError, ValueError, OSError, KeyError, TypeError) as exc:
        if not classic:
            raise
        execution_inputs, universe, settled = None, None, set()
        execution_card.update(status="skipped", blocking=[str(exc)])
    result = {"schema_version": 2, "config_digest": config.digest, "dataset_kind": config.raw["dataset_kind"],
              "status": "completed", "factors": [], "benchmark": {
                  "definition": "过滤后配置股票池等权；独立于因子和Barra覆盖",
                  "exchanges": config.exchanges,
                  **performance_metrics(benchmark["return"], annualization=config.raw["metrics"]["annualization"])},
              "sample": {"weeks": len(weeks), "periods": len(weeks),
                         "rebalance_frequency": config.rebalance_frequency,
                         "sample_policy": config.sample_policy, "signal_start": manifest["signal_start"],
                         "signal_end": manifest["signal_end"], "holding_start": manifest["holding_start"],
                         "holding_end": manifest["holding_end"]}}
    for factor in config.factors:
        signals = features[features.factor.eq(factor["name"])].copy()
        tables = evaluate_factor(signals, labels, benchmark, weeks,
                                 missing_returns=config.data_policy["missing_returns"],
                                 empty_groups=config.data_policy["empty_groups"],
                                 sample_policy=config.sample_policy, variants=config.variants,
                                 **config.raw["metrics"])
        directory = config.output / "factors" / factor["name"]
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = save_result_tables(directory, tables)
        rows = coverage[coverage.factor.eq(factor["name"])]
        ic = tables["ic_summary"]
        main = ic[ic.method.eq("spearman") & ic.subset.eq("ALL")]
        summary = {row.variant: {"mean_rank_ic": row.ic_mean, "annual_icir": row.annual_icir,
                                 "t_stat": row.t_stat, "valid_ic_weeks": row.n}
                   for row in main.itertuples(index=False)}
        card = {"name": factor["name"], "label": factor["label"],
                "output_dir": str(directory.relative_to(config.output)),
                "sample": {"weeks": len(weeks), "periods": len(weeks),
                           "rebalance_frequency": config.rebalance_frequency,
                           "sample_policy": config.sample_policy,
                           "signal_rows": len(signals.drop_duplicates(["entry", "exit", "code"])),
                           "variant_signal_rows": {v: int(signals.variant.eq(v).sum()) for v in config.variants},
                           "mean_variant_coverage": {v: float(rows[f"variant_{v}_coverage"].mean())
                                                     for v in config.variants if f"variant_{v}_coverage" in rows},
                           "mean_raw_coverage": rows.raw_coverage.mean(),
                           "mean_common_coverage": rows.common_coverage.mean(),
                           "signal_start": manifest["signal_start"], "signal_end": manifest["signal_end"],
                           "holding_start": manifest["holding_start"], "holding_end": manifest["holding_end"]},
                "summary": summary}
        card["execution"] = dict(execution_card)
        card["variants"] = config.variants
        card["sample_policy"] = config.sample_policy
        card["rebalance_frequency"] = config.rebalance_frequency
        card["preprocess"] = {"winsorize": config.raw["preprocess"].get("winsorize", True),
                              "standardize": config.raw["preprocess"].get("standardize", True)}
        card["market_cap"] = config.market_cap if config.requires_market_cap else None
        card["artifacts"] = artifacts
        card["data_policy"] = config.data_policy
        card["price_contract"] = config.raw.get("price_contract", {})
        if execution_inputs is not None:
            from .execution import run_execution
            try:
                executed = run_execution(signals, execution_inputs["execution_quotes"],
                    execution_inputs["execution_status"], execution_inputs["corporate_actions"], weeks,
                    execution_inputs["calendar"], terminal=execution_inputs["execution_terminal"],
                    initial_capital=execution_settings.get("initial_capital", 1_000_000.),
                    annualization=config.raw["metrics"]["annualization"], price_mode=execution_mode,
                    delistings=universe.drop_duplicates(["code", "delisting_date"]) if execution_mode == "adjusted_units" else None)
                if execution_mode == "raw":
                    unresolved = unresolved_held_delistings(executed["holdings"], universe, settled)
                    if unresolved:
                        raise DataGateError(f"执行账本有 {len(unresolved)} 个实际持有至退市的代码缺少已验证退出事件；"
                                            "退市日期不是现金结算，不能猜测兑付或将其永久按旧价估值")
            except (DataGateError, ValueError, OSError, KeyError, TypeError) as exc:
                if not classic:
                    raise
                card["execution"].update(status="skipped", blocking=[str(exc)])
                executed = None
        else:
            executed = None
        if executed is not None:
            for name, table in executed.items():
                if name in ("holdings", "trades", "daily"):
                    table.to_parquet(directory / f"execution_{name}.parquet", index=False)
                else:
                    table.to_csv(directory / f"execution_{name}.csv", index=False)
                    if name == "weekly":
                        table.to_csv(directory / "execution_periodic.csv", index=False)
            incomplete = int(executed["weekly"].quality_status.ne("ok").sum())
            definition = ("复权价格单位、代理成交限制、现金约束、费税与冲击为零；单位数非实际股数，"
                          "不重复记分红送转，实际持有退市未解决的账户收益记缺失；附加敏感性账本"
                          if execution_mode == "adjusted_units" else
                          "原始开盘价、显式公司行动、现金约束、分数股、费税与冲击为零；附加成交限制敏感性账本")
            card["execution"].update(enabled=True, status="completed", initial_capital=execution_settings.get("initial_capital", 1_000_000.),
                price_mode=execution_mode, quality_status="incomplete" if incomplete else "ok",
                incomplete_account_weeks=incomplete, definition=definition)
        write_json(directory / "summary.json", card)
        result["factors"].append(card)
    # No claim of an unseen holdout: this is a descriptive factor test of the stated history.
    with (config.output / "experiment_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                 "schema_version": 2, "config_digest": config.digest,
                                 "dataset_kind": config.raw["dataset_kind"],
                                 "factor_names": [f["name"] for f in config.factors]}, ensure_ascii=False) + "\n")
    write_json(config.output / "results.json", result)
    # Return the exact JSON-safe representation, including nulls instead of NaN.
    return json.loads((config.output / "results.json").read_text(encoding="utf-8"))
