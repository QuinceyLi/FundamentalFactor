"""Prepare explicitly labelled local research proxies without overwriting inputs."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fundamental_factor.proxy_inputs import prepare_proxy_inputs, refresh_trade_status

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--st", type=Path, default=Path("/data/mongo/public_st_history_szse.feather"))
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/proxy_inputs"))
    args = parser.parse_args()
    if args.status_only:
        result = refresh_trade_status(args.output.resolve(), Path("outputs/gfoptim_inputs/raw_quotes.parquet").resolve(),
            Path("/data/mongo/public_stock_master_sh_sz.feather"), Path("/data/mongo/public_trade_calendar_sina.feather"))
    else:
        result = prepare_proxy_inputs(args.output.resolve(), Path("outputs/gfoptim_inputs/raw_quotes.parquet").resolve(),
            Path("/data/mongo/on_all_temp.feather"), args.st.resolve(),
            Path("/data/mongo/public_stock_master_sh_sz.feather"), Path("/data/mongo/public_trade_calendar_sina.feather"))
    print({key: result[key] for key in ("rows", "codes", "st_unknown_fraction", "limit_unknown_rows")})
