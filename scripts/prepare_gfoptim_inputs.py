"""Export reusable GFOptim inputs and a candid data-gap report; sources are read-only."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fundamental_factor.gf_adapter import prepare_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/gfoptim_inputs"))
    parser.add_argument("--old-quotes", type=Path, default=Path("/home/liqinghua/data/quote.feather"))
    parser.add_argument("--recent-quotes", type=Path, default=Path("/home/liqinghua/data/quote_2024-2026.feather"))
    parser.add_argument("--exposures", type=Path, default=Path("/home/liqinghua/data/combined_exp.feather"))
    parser.add_argument("--actions", type=Path, default=Path("/home/liqinghua/GFOptim/min_te/results/alldiscount_audit_inputs/corporate_actions_20250101_20260831.csv"))
    parser.add_argument("--start", default="2013-01-01")
    args = parser.parse_args()
    result = prepare_inputs(args.output.resolve(), args.old_quotes, args.recent_quotes, args.exposures, args.actions, args.start)
    print(f"已导出 {result['quotes']['rows']:,} 条原始行情；正式门槛仍未通过。")
    print(args.output.resolve() / "data_requirements.html")
