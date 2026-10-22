from __future__ import annotations

import argparse
import sys

from .audit import audit
from .config import DataGateError, load_config
from .data import write_json
from .panel import build_panel
from .report import report
from .research import run_research


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="单因子周度分组回测与标准测试报告")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["audit", "build-panel", "run"]:
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    command = commands.add_parser("report")
    command.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        try:
            print(report(args.run_dir))
            return 0
        except (ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    config = None
    try:
        config = load_config(args.config)
        if args.command == "audit":
            result = audit(config)
            print(report(config.output))
            for issue in result["blocking"]:
                print(issue, file=sys.stderr)
            return 0 if result["gate_passed"] else 2
        if args.command == "build-panel":
            result = build_panel(config)
            print(f"面板状态：{result['status']}")
        else:
            result = run_research(config)
            print(f"已完成 {len(result['factors'])} 个因子的 {len(config.variants)} 套信号测试")
        failure_path = config.output / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
        print(report(config.output))
        return 0
    except DataGateError as exc:
        print(f"核心输入或回测一致性检查未通过：{exc}", file=sys.stderr)
        if config:
            write_json(config.output / "failure.json", {"command": args.command, "message": str(exc), "kind": "data_gate"})
            print(report(config.output))
        return 2
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"执行失败：{exc}", file=sys.stderr)
        if config:
            write_json(config.output / "failure.json", {"command": args.command, "message": str(exc), "kind": "execution_error"})
            print(report(config.output))
        return 1


if __name__ == "__main__":
    sys.exit(main())
