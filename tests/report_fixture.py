"""Small frozen runs for the unified report contract, without production data."""
import hashlib
import json
from pathlib import Path

from fundamental_factor.backtest import evaluate_factor
from fundamental_factor.report import report
from tests.test_backtest_edges import _fixture


def settings():
    return {"schema_version": 2, "dataset_kind": "synthetic",
            "factors": [{"name": "ROIC_new", "label": "ROIC"}],
            "backtest": {"rebalance_frequency": "weekly"},
            "metrics": {"annualization": 52, "nw_lag": 13},
            "preprocess": {"winsorize": False, "standardize": False},
            "neutralization": {"variants": ["raw"], "sample_policy": "independent",
                               "include_float_mcap": False},
            "data_policy": {"mode": "classic", "missing_returns": "available_case", "empty_groups": "cash"}}


def write_json(root, name, value):
    (Path(root) / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def create_run(root, config=None, tables=None, *, audit=None, quality=None, execution=None):
    root = Path(root)
    config = config or settings()
    folder = root / "factors" / "ROIC_new"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    metadata = {
        "config.frozen.json": config,
        "audit.json": {"config_digest": digest, "gate_passed": True, "sources": {}, **(audit or {})},
        "panel_manifest.json": {"config_digest": digest, "status": "ready", **(quality or {})},
        "results.json": {"config_digest": digest, "schema_version": 2, "dataset_kind": config["dataset_kind"],
                         "status": "completed", "factors": [{"name": "ROIC_new", "label": "ROIC",
                         "output_dir": "factors/ROIC_new", "execution": execution or {"enabled": False}}]},
    }
    for name, value in metadata.items():
        write_json(root, name, value)
    if tables is None:
        tables = evaluate_factor(*_fixture(first_low=.01, first_high=.1),
                                 missing_returns="available_case", empty_groups="cash")
    for name, table in tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
    return folder


def render(root):
    path = report(root)
    return json.loads((Path(root) / "report_data.json").read_text()), path.read_text()
