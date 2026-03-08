#!/usr/bin/env python3
"""Run Phase-1 aux stabilization sweep (A0 + aux-only grid).

This script follows the prioritized plan in docs/ablation_next_steps.md:
- keep shaping/hard/focus/clash disabled (A0 scaffold)
- sweep only lambda_cos and lambda_mag at conservative scales
- optionally set aux warmup epochs for gradual activation

Usage:
  python scripts/run_aux_stabilization_sweep.py --seed 42
  python scripts/run_aux_stabilization_sweep.py --seed 42 --warmup-epochs 10 --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import subprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run aux-only stabilization sweep on top of A0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--python", default="python")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--extra-override", action="append", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    lambda_cos_grid = [0.05, 0.1, 0.2]
    lambda_mag_grid = [0.01, 0.05, 0.1]

    baseline_off = {
        "model.mse_weight_peak": 1.0,
        "model.mse_weight_min": 1.0,
        "model.mse_hard_beta": 0.0,
        "model.disp_focus_weight": 1.0,
        "model.disp_outside_focus_weight": 1.0,
        "model.lambda_clash": 0.0,
    }

    for idx, (lambda_cos, lambda_mag) in enumerate(itertools.product(lambda_cos_grid, lambda_mag_grid), start=1):
        run_id = f"AUX{idx:02d}"
        cmd = [
            args.python,
            "train.py",
            f"seed={args.seed}",
            f"study_name=aux_stabilization/{run_id}_seed{args.seed}",
            f"model.lambda_cos={lambda_cos}",
            f"model.lambda_mag={lambda_mag}",
            f"model.cos_warmup_epochs={args.warmup_epochs}",
            f"model.mag_warmup_epochs={args.warmup_epochs}",
            "model.hard_warmup_epochs=0",
            "model.focus_warmup_epochs=0",
        ]
        cmd.extend(f"{k}={v}" for k, v in baseline_off.items())
        cmd.extend(args.extra_override)

        print(f"[{run_id}] lambda_cos={lambda_cos}, lambda_mag={lambda_mag}")
        print("  " + " ".join(shlex.quote(x) for x in cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
