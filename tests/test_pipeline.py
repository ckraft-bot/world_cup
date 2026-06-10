from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest and store timestamped test logs."
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Optional additional pytest arguments, for example: -q -k data_prep",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    logs_dir = project_dir / "tests" / "test_results"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"pytest_{timestamp}.log"
    
    command = [sys.executable, "-m", "pytest"]
    if args.pytest_args:
        command.extend(args.pytest_args)
    else:
        command.append("-q")

    print(f"Running: {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    output = result.stdout
    if result.stderr:
        output = f"{output}\n{result.stderr}" if output else result.stderr

    print(output)

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(output)
        handle.write(f"\nExit code: {result.returncode}\n")

    print(f"Saved test log: {log_path}")
    print(f"Exit code: {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
