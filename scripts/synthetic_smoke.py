"""Generate clearly labelled synthetic factor reports; never reads /data/mongo."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fundamental_factor.config import load_config
from fundamental_factor.report import report
from fundamental_factor.research import run_research
from tests.fixtures import disk_fixture, enable_execution_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/factor_demo")
    parser.add_argument("--with-execution", action="store_true", help="Include the constrained execution ledger")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    if (root / "config.json").exists():
        config = load_config(root / "config.json")
        if config.raw.get("dataset_kind") != "synthetic":
            parser.error("Existing directory is not a synthetic fixture")
    else:
        config = disk_fixture(root)
    if args.with_execution and not config.raw.get("execution", {}).get("enabled"):
        if (config.output / "config.frozen.json").exists():
            parser.error("请为执行账本演示指定新的 --output-root，避免修改冻结规格")
        config = enable_execution_fixture(config)
    result = run_research(config)
    print("合成数据验收完成（不代表真实因子表现）:", result["status"])
    print(report(config.output))


if __name__ == "__main__":
    main()
