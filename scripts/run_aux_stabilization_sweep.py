#!/usr/bin/env python3
"""Run Phase-1 aux stabilization sweep (A0 + aux-only grid).

This script follows the prioritized plan in docs/ablation_next_steps.md:
- keep focus/clash disabled (A0 scaffold)
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
from typing import Sequence


def _parse_float_list(raw: str) -> list[float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("grid must include at least one numeric value")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list: {raw}") from exc


def _fmt_float(value: float) -> str:
    return f"{value:g}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run aux-only stabilization sweep on top of A0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--python", default="python")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument(
        "--lambda-cos-grid",
        type=_parse_float_list,
        default=[0.05, 0.1, 0.2],
        help="Comma-separated lambda_cos sweep values (default: 0.05,0.1,0.2)",
    )
    p.add_argument(
        "--lambda-mag-grid",
        type=_parse_float_list,
        default=[0.01, 0.05, 0.1],
        help="Comma-separated lambda_mag sweep values (default: 0.01,0.05,0.1)",
    )
    p.add_argument(
        "--study-prefix",
        default="aux_stabilization",
        help="Hydra study_name prefix for runs",
    )
    p.add_argument(
        "--run-prefix",
        default="AUX",
        help="Prefix for per-run labels in study_name",
    )
    p.add_argument("--extra-override", action="append", default=[])
    return p.parse_args()


def _build_command(
    *,
    python_bin: str,
    seed: int,
    run_label: str,
    study_prefix: str,
    lambda_cos: float,
    lambda_mag: float,
    warmup_epochs: int,
    baseline_off: dict[str, float],
    extra_overrides: Sequence[str],
) -> list[str]:
    cmd = [
        python_bin,
        "train.py",
        f"seed={seed}",
        f"study_name={study_prefix}/{run_label}_seed{seed}",
        f"model.lambda_cos={_fmt_float(lambda_cos)}",
        f"model.lambda_mag={_fmt_float(lambda_mag)}",
        f"model.cos_warmup_epochs={warmup_epochs}",
        f"model.mag_warmup_epochs={warmup_epochs}",
        "model.focus_warmup_epochs=0",
    ]
    cmd.extend(f"{k}={_fmt_float(v)}" for k, v in baseline_off.items())
    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args = parse_args()

    baseline_off = {
        # Keep base MSE weighting neutral outside of the swept aux terms.
        "model.disp_focus_weight": 1.0,
        "model.disp_over_max_weight": 1.0,
        "model.lambda_clash": 0.0,
        "model.lambda_high_plddt_l2": 0.0,
        "model.lambda_low_plddt_l2": 0.0,
    }

    total_runs = len(args.lambda_cos_grid) * len(args.lambda_mag_grid)
    print(f"Planned runs: {total_runs}")

    for idx, (lambda_cos, lambda_mag) in enumerate(
        itertools.product(args.lambda_cos_grid, args.lambda_mag_grid), start=1
    ):
        run_id = f"{args.run_prefix}{idx:02d}"
        cmd = _build_command(
            python_bin=args.python,
            seed=args.seed,
            run_label=run_id,
            study_prefix=args.study_prefix,
            lambda_cos=lambda_cos,
            lambda_mag=lambda_mag,
            warmup_epochs=args.warmup_epochs,
            baseline_off=baseline_off,
            extra_overrides=args.extra_override,
        )

        print(f"[{run_id}] lambda_cos={lambda_cos}, lambda_mag={lambda_mag}")
        print("  " + " ".join(shlex.quote(x) for x in cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
