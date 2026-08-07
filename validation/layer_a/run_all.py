#!/usr/bin/env python3
"""Run every Layer A script, preserve all diagnostics, and aggregate results."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import FAIL, aggregate_results

SCRIPTS = (
    "check_environment_conventions.py",
    "check_synthetic_ground_truth.py",
    "check_dare_lyapunov.py",
    "check_var_ss_conversions.py",
    "check_autocovariances.py",
    "check_transfer_spectra.py",
    "check_installed_repo_triplets.py",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()

    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parents[1]
    args.output.mkdir(parents=True, exist_ok=True)

    execution: list[dict[str, object]] = []
    for script_name in SCRIPTS:
        command = [
            sys.executable,
            str(script_dir / script_name),
            "--output",
            str(args.output),
        ]
        print(f"\n=== Running {script_name} ===", flush=True)
        completed = subprocess.run(command, cwd=repository_root, check=False)
        execution.append(
            {
                "script": script_name,
                "returncode": completed.returncode,
                "command": command,
            }
        )

    (args.output / "execution.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )
    result_paths = sorted(
        path
        for path in args.output.glob("*.json")
        if path.name
        not in {
            "environment.json",
            "execution.json",
            "synthetic_ground_truth_metadata.json",
            "layer_a_summary.json",
            "installed_repo_triplets.json",
        }
    )
    rows = aggregate_results(result_paths, args.output)
    failed_rows = [row for row in rows if row.get("status") == FAIL]
    crashed = [item for item in execution if item["returncode"] != 0]

    print(
        f"\nLayer A summary: {len(rows)} checks, "
        f"{len(failed_rows)} classified failures, {len(crashed)} non-zero scripts."
    )
    return 1 if failed_rows or crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())
