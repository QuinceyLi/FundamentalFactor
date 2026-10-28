"""Prepare an independent AIQ-filled status source while reusing audited inputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyarrow.compute as pc
import pyarrow.feather as feather

from fundamental_factor.aiq_status import prepare_aiq_matrix_status
from fundamental_factor.data import dates, fingerprint, write_json


def prepare(base_manifest: Path, output: Path, *, stock_batch_size=128) -> dict:
    base_manifest, output = base_manifest.resolve(), output.resolve()
    base_root = base_manifest.parent
    original_manifest = fingerprint(base_manifest)
    previous = json.loads(base_manifest.read_text(encoding="utf-8"))
    aiq_root = Path(previous["aiq_root"]).resolve()
    protected_path = Path(previous["price_source_protected"]["path"]).resolve()
    if output.is_relative_to(base_root) or any(output.is_relative_to(path) for path in (aiq_root, protected_path.parent)):
        raise ValueError("New matrix artifacts must be separate from sources and the frozen preparation directory")
    protected = fingerprint(protected_path)
    if protected != previous["price_source_protected"]:
        raise ValueError("Protected adjusted prices no longer match the prior audited fingerprint")
    date_column = feather.read_table(protected_path, columns=["date"], memory_map=True)["date"]
    bounds = pc.min_max(date_column).as_py()
    endpoints = dates(pd.Series([bounds["min"], bounds["max"]]))
    output.mkdir(parents=True, exist_ok=True)
    summary = copy.deepcopy(previous)
    summary.update({"prepared_directory": str(output), "base_preparation_manifest": original_manifest,
                    "missing_values": {"method": "aiq_matrix"}, "source_files_modified": False,
                    "aiq_modules_executed": False})
    summary["datasets"]["status"] = prepare_aiq_matrix_status(
        aiq_root / "database/quote/quote.feather", base_root / "trade_calendar.parquet",
        base_root / "stock_master.parquet", base_root / "st_history.parquet", output / "trade_status.parquet",
        start=endpoints.iloc[0], end=endpoints.iloc[1], stock_batch_size=stock_batch_size)
    summary["source_selection"]["status"] = (
        "AIQ原始行情按交易日与证券生命周期展开；open保持缺失，close/last前填，volume首个有效值后缺失补0，"
        "再计算GF停牌与现有开盘限制代理；保留原始值、缺失/填充标记和实际供值日；ST未知不泛填")
    summary["matrix_fill_scope"] = {
        "status": "Prepared daily raw quote matrix and proxies in this manifest",
        "factors_and_controls": "Filled at panel construction under the explicit aiq_matrix config, with original values retained",
        "return_prices": "Protected repaired price source unchanged; endpoint matrix fill is recorded in label provenance",
        "st": "Existing verified/proxy state events reused, including explicit unknown coverage expiry"}
    summary["remaining_improvements"] = [item.replace(
        "未知不自动补零或变成有效控制", "矩阵填充不能认证原始数据完整，零/负/无穷原值仍不能成为有效市值控制")
        for item in summary.get("remaining_improvements", [])]
    if fingerprint(protected_path) != protected or fingerprint(base_manifest) != original_manifest:
        raise RuntimeError("Protected price or frozen preparation manifest changed")
    summary["price_source_protected"] = protected
    write_json(output / "manifest.json", summary)
    print(output / "manifest.json", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=Path("outputs/aiq_inputs_20261028/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/aiq_matrix_inputs_20261028"))
    parser.add_argument("--stock-batch-size", type=int, default=128)
    args = parser.parse_args()
    prepare(args.base_manifest, args.output, stock_batch_size=args.stock_batch_size)
