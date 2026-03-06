#!/usr/bin/env python3
"""Run all phase-1 disp_1to5 ablation plan commands.

Usage:
  python scripts/run_disp1to5_ablation_all.py
  python scripts/run_disp1to5_ablation_all.py --seed 1337
  python scripts/run_disp1to5_ablation_all.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Allow importing sibling script as a module.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from disp1to5_ablation import default_matrix, hydra_cmd  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all phase-1 disp_1to5 ablation commands")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for all runs (default: 42)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all commands without executing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = default_matrix()

    for spec in runs:
        cmd = hydra_cmd(spec, seed=args.seed)
        print(f"[{spec.run_id}] {cmd}")
        if args.dry_run:
            continue

        subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
