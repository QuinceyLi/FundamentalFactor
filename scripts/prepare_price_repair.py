"""Stage an audited repaired copy; never overwrite /data/mongo source files.

The optional public fetch is read-only and archives exact bytes plus SHA256.
Without --fetch-public, preparation is reproducible entirely from saved inputs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fundamental_factor.data import write_json
from fundamental_factor.price_repair import prepare_repaired_prices


def fetch_public(output: Path) -> None:
    import requests

    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    common = {"fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
              "klt": "101", "secid": "0.000001", "beg": "20250520", "end": "20250620"}
    endpoint = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    requests_to_make = [
        ("sina_000001_dividends.html", "https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/000001.phtml", {}),
        ("eastmoney_000001_raw.json", endpoint, {**common, "fqt": "0"}),
        ("eastmoney_000001_hfq.json", endpoint, {**common, "fqt": "2"}),
    ]
    records = []
    for filename, url, params in requests_to_make:
        response = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        (raw / filename).write_bytes(response.content)
        records.append({"file": filename, "url": response.url, "status": response.status_code,
            "sha256": hashlib.sha256(response.content).hexdigest(), "bytes": len(response.content),
            "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    write_json(raw / "fetch_manifest.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/price_repair"))
    parser.add_argument("--mongo", type=Path, default=Path("/data/mongo/quotefac.feather"))
    parser.add_argument("--legacy", type=Path, default=Path("/home/liqinghua/data/quote.feather"))
    parser.add_argument("--reference", type=Path, default=Path("/home/liqinghua/data/quote_2024-2026.feather"))
    parser.add_argument("--fetch-public", action="store_true")
    arguments = parser.parse_args()
    if arguments.fetch_public:
        fetch_public(arguments.output)
    result = prepare_repaired_prices(arguments.output, arguments.mongo, arguments.legacy, arguments.reference)
    print(json.dumps({"adjusted_rows": result["adjusted_rows"], "patch": result["patch"],
        "event_close_return_before": result["event_close_return_before"],
        "event_close_return_after": result["event_close_return_after"],
        "output": str(arguments.output / "validated")}, ensure_ascii=False, indent=2))
