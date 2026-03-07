#!/usr/bin/env python3
"""Run additive/constructive ablation experiments (A0 -> F1).

Each experiment starts from the fully-disabled baseline (A0) and restores exactly
one logical loss group to its default value.

Usage:
  python scripts/run_additive_ablation_all.py
  python scripts/run_additive_ablation_all.py --seed 1337 --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentSpec:
    run_id: str
    tag: str
    purpose: str
    overrides: dict[str, float] = field(default_factory=dict)


# A0: fully disabled baseline for the listed loss groups.
BASELINE_OFF_OVERRIDES: dict[str, float] = {
    "model.mse_weight_peak": 1.0,
    "model.mse_weight_min": 1.0,
    "model.mse_hard_beta": 0.0,
    "model.disp_focus_weight": 1.0,
    "model.disp_outside_focus_weight": 1.0,
    "model.lambda_cos": 0.0,
    "model.lambda_mag": 0.0,
    "model.lambda_clash": 0.0,
}


def additive_matrix() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            run_id="A0",
            tag="baseline",
            purpose="Vanilla baseline with shaping/hard-mining/focus/aux/clash disabled.",
            overrides={},
        ),
        ExperimentSpec(
            run_id="B1",
            tag="+shaping",
            purpose="Restore only MSE shaping (peak=2.0, min=0.5).",
            overrides={
                "model.mse_weight_peak": 2.0,
                "model.mse_weight_min": 0.5,
            },
        ),
        ExperimentSpec(
            run_id="C1",
            tag="+hard",
            purpose="Restore only hard-example mining (beta=3.0).",
            overrides={
                "model.mse_hard_beta": 3.0,
            },
        ),
        ExperimentSpec(
            run_id="D1",
            tag="+focus",
            purpose="Restore only spatial focus for 1-5Å (focus=1.5, outside=0.9).",
            overrides={
                "model.disp_focus_weight": 1.5,
                "model.disp_outside_focus_weight": 0.9,
            },
        ),
        ExperimentSpec(
            run_id="E1",
            tag="+aux",
            purpose="Restore only geometric auxiliary losses (lambda_cos=1.0, lambda_mag=1.0).",
            overrides={
                "model.lambda_cos": 1.0,
                "model.lambda_mag": 1.0,
            },
        ),
        ExperimentSpec(
            run_id="F1",
            tag="+clash",
            purpose="Restore only physical anti-clash regularization (lambda_clash=0.05).",
            overrides={
                "model.lambda_clash": 0.05,
            },
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run additive loss-group ablations (A0, B1, C1, D1, E1, F1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for all runs (default: 42)")
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable used to launch train.py (default: python)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help='Extra Hydra override appended to every run (repeatable), e.g. --extra-override "trainer.max_epochs=20"',
    )
    return parser.parse_args()


def build_command(spec: ExperimentSpec, seed: int, python_bin: str, extra_overrides: list[str]) -> list[str]:
    merged = dict(BASELINE_OFF_OVERRIDES)
    merged.update(spec.overrides)

    cmd = [
        python_bin,
        "train.py",
        f"seed={seed}",
        f"study_name=additive_ablation/{spec.run_id}_seed{seed}",
    ]
    for key, value in merged.items():
        cmd.append(f"{key}={value}")

    cmd.extend(extra_overrides)
    return cmd


def main() -> None:
    args = parse_args()

    for spec in additive_matrix():
        cmd = build_command(spec, args.seed, args.python, args.extra_override)
        quoted = " ".join(shlex.quote(part) for part in cmd)

        print(f"[{spec.run_id} {spec.tag}] {spec.purpose}")
        print(f"  {quoted}")

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
