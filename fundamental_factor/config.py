from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


class DataGateError(RuntimeError):
    """Inputs are not sufficient for publishing a formal backtest."""


RISK_FIELDS = [
    "AnalystSentiment", "BETA", "BTOP", "DividendYield", "EarningQuality",
    "EarningsYield", "EarningVariability", "GROWTH", "INDMOM", "InvestmentQuality",
    "LEVERAGE", "LIQUIDTY", "LongTermReversal", "MOMENTUM", "Profitability",
    "RESVOL", "SEASON", "SIZE", "STREV", "MIDCAP",
]
VARIANTS = ["raw", "industry_size", "full"]
DEFAULT_BACKTEST = {"min_listing_sessions": 120, "start_date": None, "end_date": None}
DEFAULT_PREPROCESS = {"mad_multiplier": 5., "mad_normalizer": 1.4826,
                      "fallback_quantiles": [.01, .99], "rank_tolerance": 1e-10}
DEFAULT_METRICS = {"annualization": 52, "nw_lag": 13}


@dataclass(frozen=True)
class Config:
    raw: dict
    path: Path

    @property
    def output(self) -> Path:
        return Path(self.raw["output_dir"])

    @property
    def factors(self) -> list[dict]:
        return self.raw["factors"]

    @property
    def include_float_mcap(self) -> bool:
        # Do not inject this default into raw: existing frozen digests stay valid.
        return self.raw.get("neutralization", {}).get("include_float_mcap", True)

    @property
    def exchanges(self) -> list[str] | None:
        return self.raw.get("universe", {}).get("exchanges")

    @property
    def variants(self) -> list[str]:
        default = ["raw", "industry_size" if self.include_float_mcap else "industry", "full"]
        return list(self.raw.get("neutralization", {}).get("variants", default))

    @property
    def sample_policy(self) -> str:
        # Missing settings retain the sample definition of existing frozen runs.
        return self.raw.get("neutralization", {}).get("sample_policy", "common")

    @property
    def rebalance_frequency(self) -> str:
        return self.raw.get("backtest", {}).get("rebalance_frequency", "weekly")

    @property
    def optional_control_sources(self) -> bool:
        return (self.data_policy["mode"] == "classic" and self.sample_policy == "independent"
                and "raw" in self.variants)

    @property
    def requires_industry(self) -> bool:
        return any(variant in {"industry", "industry_size", "full"} for variant in self.variants)

    @property
    def requires_risk(self) -> bool:
        return "full" in self.variants

    @property
    def requires_float_mcap(self) -> bool:
        return self.requires_market_cap and self.market_cap["method"] == "history"

    @property
    def requires_market_cap(self) -> bool:
        return any(variant in {"size", "industry_size"} for variant in self.variants) or (
            "full" in self.variants and self.include_float_mcap)

    @property
    def market_cap(self) -> dict:
        # Resolve defaults here, preserving frozen configurations and their digests.
        return {"method": "history", "quotes_source": "market_cap_quotes", "turnover_source": "quotes",
                "close_column": "close", "volume_column": "volume", "turnover_column": "turnover",
                "volume_unit": "lot", "turnover_unit": "percent", "price_basis": "unadjusted"} | (
                    self.raw.get("neutralization", {}).get("market_cap", {}))

    @property
    def market_cap_column(self) -> str:
        return "market_cap_proxy" if self.market_cap["method"] == "quote_proxy" else "float_mcap"

    @property
    def uses_market_cap_proxy(self) -> bool:
        return self.requires_market_cap and self.market_cap["method"] == "quote_proxy"

    @property
    def data_policy(self) -> dict:
        """Classic analytics is the default; explicit legacy policies remain reproducible."""
        supplied = self.raw.get("data_policy", {})
        mode = supplied.get("mode", "classic")
        defaults = ({"st_unknown": "include", "suspension_unknown": "include", "missing_returns": "report"}
                    if mode == "classic" else
                    {"st_unknown": "error", "suspension_unknown": "error", "missing_returns": "error"})
        return {"mode": mode, **defaults, "empty_groups": "missing", "description": ""} | supplied

    @property
    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()

    def source(self, name: str) -> Path | None:
        value = self.raw["sources"].get(name)
        return Path(value) if value else None


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 2 or "study" in raw:
        raise ValueError("需要 schema_version=2 的单因子配置；旧训练配置不兼容，请参考 configs/factor_test.json")
    for obsolete in ("training_weeks", "validation_weeks", "holdout_weeks", "model"):
        if obsolete in raw:
            raise ValueError(f"单因子框架不接受训练参数：{obsolete}")
    base = path.parent
    root = (base / raw.get("data_root", ".")).resolve()
    raw["data_root"] = str(root)
    raw["output_dir"] = str((base / raw.get("output_dir", "../outputs/factor_test")).resolve())
    raw.setdefault("dataset_kind", "observed")
    if raw["dataset_kind"] not in ("observed", "synthetic"):
        raise ValueError("dataset_kind must be observed or synthetic")
    if not isinstance(raw.get("sources"), dict) or not raw.get("factors"):
        raise ValueError("sources 和非空 factors 列表必须配置")
    for key, value in raw["sources"].items():
        if value:
            raw["sources"][key] = str((root / value).resolve())
    names = set()
    for factor in raw["factors"]:
        name = factor.get("name", "")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in names:
            raise ValueError("factor.name 必须是唯一的字母/数字/下划线标识")
        if not factor.get("column") or factor.get("source") not in raw["sources"]:
            raise ValueError(f"因子 {name} 缺少 column 或有效 source 映射")
        factor.setdefault("label", name)
        names.add(name)
    frequency = raw.get("backtest", {}).get("rebalance_frequency", "weekly")
    if frequency not in ("weekly", "monthly"):
        raise ValueError("backtest.rebalance_frequency 必须为 weekly 或 monthly")
    metric_defaults = DEFAULT_METRICS if frequency == "weekly" else {"annualization": 12, "nw_lag": 3}
    extra_options = {"backtest": {"rebalance_frequency"}, "preprocess": {"winsorize", "standardize"}}
    for key, defaults in (("backtest", DEFAULT_BACKTEST), ("preprocess", DEFAULT_PREPROCESS),
                          ("metrics", metric_defaults)):
        supplied = raw.get(key, {})
        allowed = set(defaults) | extra_options.get(key, set())
        if set(supplied) - allowed:
            raise ValueError(f"未知 {key} 参数：{sorted(set(supplied) - allowed)}")
        raw[key] = defaults | supplied
    if "rebalance_frequency" in raw["backtest"] and raw["metrics"]["annualization"] != metric_defaults["annualization"]:
        raise ValueError(f"{frequency} 调仓的 metrics.annualization 必须为 {metric_defaults['annualization']}")
    for key, value in (("min_listing_sessions", raw["backtest"]["min_listing_sessions"]),
                       ("annualization", raw["metrics"]["annualization"]),
                       ("nw_lag", raw["metrics"]["nw_lag"])):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    pre = raw["preprocess"]
    for flag in ("winsorize", "standardize"):
        if flag in pre and not isinstance(pre[flag], bool):
            raise ValueError(f"preprocess.{flag} must be boolean")
    for key in ("mad_multiplier", "mad_normalizer", "rank_tolerance"):
        if not isinstance(pre[key], (int, float)) or not 0 < pre[key] < float("inf"):
            raise ValueError(f"preprocess.{key} must be finite and positive")
    q = pre["fallback_quantiles"]
    if not isinstance(q, list) or len(q) != 2 or not 0 <= q[0] < q[1] <= 1:
        raise ValueError("fallback_quantiles must contain two increasing probabilities")
    if raw.get("risk_fields", RISK_FIELDS) != RISK_FIELDS:
        raise ValueError("全中性使用冻结的20项Barra暴露；risk_fields 不可静默增删")
    raw["risk_fields"] = list(RISK_FIELDS)
    raw.setdefault("price_contract", {})
    if "data_policy" in raw:
        policy = raw["data_policy"]
        choices = {"mode": {"classic", "strict", "proxy"}, "st_unknown": {"error", "include", "exclude"},
                   "suspension_unknown": {"error", "include", "exclude"},
                   "missing_returns": {"error", "report", "available_case"},
                   "empty_groups": {"missing", "cash"}}
        if not isinstance(policy, dict) or set(policy) - (set(choices) | {"description"}):
            raise ValueError("data_policy 包含未知参数")
        for key, values in choices.items():
            if key in policy and policy[key] not in values:
                raise ValueError(f"无效的 data_policy.{key}")
        if policy.get("mode", "classic") == "strict" and any(
                policy.get(key, "error") != "error" for key in ("st_unknown", "suspension_unknown", "missing_returns")):
            raise ValueError("代理或不完整标签处理必须声明 data_policy.mode=proxy 或 classic")
        if policy.get("mode") == "proxy" and not isinstance(policy.get("description"), str):
            raise ValueError("proxy data_policy 需要 description 说明")
    if "neutralization" in raw:
        settings = raw["neutralization"]
        if (not isinstance(settings, dict) or not settings or set(settings) - {"include_float_mcap", "variants", "sample_policy", "market_cap"}
                or "include_float_mcap" in settings and not isinstance(settings["include_float_mcap"], bool)):
            raise ValueError("neutralization 仅支持 include_float_mcap、variants、sample_policy 与 market_cap")
        if settings.get("sample_policy", "common") not in ("common", "independent"):
            raise ValueError("neutralization.sample_policy 必须为 common 或 independent")
        if "variants" in settings:
            variants = settings["variants"]
            if (not isinstance(variants, list) or not variants or
                    any(value not in {"raw", "size", "industry", "industry_size", "full"} for value in variants)
                    or len(set(variants)) != len(variants)):
                raise ValueError("neutralization.variants 必须为非空、无重复的有效版本列表")
        if "market_cap" in settings:
            cap = settings["market_cap"]
            allowed = {"method", "quotes_source", "turnover_source", "close_column", "volume_column",
                       "turnover_column", "volume_unit", "turnover_unit", "price_basis"}
            if not isinstance(cap, dict) or set(cap) - allowed:
                raise ValueError("neutralization.market_cap 包含未知参数")
            if cap.get("method", "history") not in {"history", "quote_proxy"}:
                raise ValueError("neutralization.market_cap.method 必须为 history 或 quote_proxy")
            for key, expected in (("volume_unit", "lot"), ("turnover_unit", "percent"),
                                  ("price_basis", "unadjusted")):
                if cap.get(key, expected) != expected:
                    raise ValueError(f"neutralization.market_cap.{key} 必须为 {expected}")
            for key in ("quotes_source", "turnover_source", "close_column", "volume_column", "turnover_column"):
                if key in cap and (not isinstance(cap[key], str) or not cap[key].strip()):
                    raise ValueError(f"neutralization.market_cap.{key} 必须为非空字段或来源名称")
            if cap.get("close_column", "close").lower() in {"acc_close", "acc_open"}:
                raise ValueError("市值代理必须使用未复权 close，不能使用 acc_close/acc_open")
            proxy_columns = [cap.get("close_column", "close"), cap.get("volume_column", "volume"),
                             cap.get("turnover_column", "turnover")]
            if any(column.lower() in {"date", "code"} for column in proxy_columns):
                raise ValueError("市值代理数值字段不能使用 date/code 键列")
            if cap.get("close_column", "close") == cap.get("volume_column", "volume"):
                raise ValueError("市值代理的 close_column 与 volume_column 不可相同")
            if (cap.get("quotes_source", "market_cap_quotes") == cap.get("turnover_source", "quotes")
                    and len(set(proxy_columns)) != 3):
                raise ValueError("市值代理同源时 close_column、volume_column 与 turnover_column 必须互异")
            if cap.get("method", "history") == "quote_proxy":
                for key, default in (("quotes_source", "market_cap_quotes"), ("turnover_source", "quotes")):
                    if cap.get(key, default) not in raw["sources"]:
                        raise ValueError(f"市值代理来源 {cap.get(key, default)} 必须在 sources 中定义；缺失数据可显式设为 null")
    if "universe" in raw:
        settings = raw["universe"]
        if not isinstance(settings, dict) or set(settings) != {"exchanges"}:
            raise ValueError("universe 必须提供 exchanges 列表")
        exchanges = settings["exchanges"]
        if (not isinstance(exchanges, list) or not exchanges
                or any(value not in ("SH", "SZ", "BJ") for value in exchanges)
                or len(exchanges) != len(set(exchanges))):
            raise ValueError("universe.exchanges 必须是非空、无重复的 SH/SZ/BJ 列表")
    if "execution" in raw:
        execution = raw["execution"]
        if not isinstance(execution, dict) or not isinstance(execution.get("enabled", False), bool):
            raise ValueError("execution.enabled must be boolean")
        capital = execution.get("initial_capital", 1_000_000.)
        if isinstance(capital, bool) or not isinstance(capital, (int, float)) or not 0 < capital < float("inf"):
            raise ValueError("execution.initial_capital must be finite and positive")
        allowed = {"enabled", "initial_capital", "raw_prices_verified", "status_verified",
                   "corporate_actions_complete", "coverage_start", "coverage_end", "description", "price_mode"}
        if set(execution) - allowed:
            raise ValueError(f"未知 execution 参数：{sorted(set(execution) - allowed)}")
        if execution.get("price_mode", "raw") not in ("raw", "adjusted_units"):
            raise ValueError("execution.price_mode must be raw or adjusted_units")
        if (execution.get("price_mode") == "adjusted_units" and
                raw.get("data_policy", {}).get("mode", "classic") == "strict"):
            raise ValueError("复权单位执行账本必须声明 data_policy.mode=proxy 或 classic")
    if raw.get("classification_workbook"):
        raw["classification_workbook"] = str((base / raw["classification_workbook"]).resolve())
    config = Config(raw, path)
    for source in raw["sources"].values():
        if source and (config.output == Path(source) or config.output in Path(source).parents):
            raise ValueError("output_dir must not contain a source dataset")
    if config.output == root or root in config.output.parents:
        raise ValueError("output_dir must be outside data_root")
    return config
