#!/usr/bin/env python3
"""Run automatic parameter search for B1 (focus) and D1 (pLDDT) additive ablations.

The search starts from the A0 fully-disabled scaffold from run_additive_ablation_all.py
and sweeps candidate values for B1 and D1-specific parameters.

Usage:
  python scripts/run_b1_d1_param_search.py
  python scripts/run_b1_d1_param_search.py --group b1 --dry-run
  python scripts/run_b1_d1_param_search.py --max-runs 12 --extra-override trainer.max_epochs=20
"""

from __future__ import annotations

import argparse
import itertools
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


BASELINE_OFF_OVERRIDES: dict[str, float] = {
    "model.disp_focus_weight": 1.0,
    "model.disp_over_max_weight": 1.0,
    "model.lambda_cos": 0.0,
    "model.lambda_mag": 0.0,
    "model.lambda_clash": 0.0,
    "model.lambda_high_plddt_l2": 0.0,
    "model.lambda_low_plddt_l2": 0.0,
    "model.plddt_gate_start": 100.0,
    "model.plddt_gate_end": 100.0,
}


@dataclass(frozen=True)
class Candidate:
    group: str
    run_id: str
    overrides: dict[str, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter search for B1 and D1 additive ablations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--python", default="python")
    p.add_argument("--group", choices=["all", "b1", "d1"], default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-runs", type=int, default=0, help="Limit total runs (0 means all candidates)")
    p.add_argument(
        "--metric",
        choices=["disp1to5", "flex"],
        default="disp1to5",
        help="Validation metric used to pick best setting",
    )
    p.add_argument("--extra-override", action="append", default=[])
    return p.parse_args()


def build_candidates(selected_group: str) -> list[Candidate]:
    candidates: list[Candidate] = []

    if selected_group in {"all", "b1"}:
        focus_weight_grid = [1.1, 1.25, 1.5, 1.75, 2.0]
        over_max_weight_grid = [0.25, 0.5, 0.75, 1.0]
        for idx, (focus_w, over_w) in enumerate(itertools.product(focus_weight_grid, over_max_weight_grid), start=1):
            candidates.append(
                Candidate(
                    group="b1",
                    run_id=f"B1S{idx:02d}",
                    overrides={
                        "model.disp_focus_weight": focus_w,
                        "model.disp_over_max_weight": over_w,
                    },
                )
            )

    if selected_group in {"all", "d1"}:
        high_l2_grid = [0.25, 0.5, 0.75, 1.0]
        low_l2_grid = [0.25, 0.5, 0.75]
        gate_start_grid = [70.0, 75.0, 80.0, 85.0]
        gate_end_grid = [95.0, 100.0]
        for idx, (high_l2, low_l2, gate_start, gate_end) in enumerate(
            itertools.product(high_l2_grid, low_l2_grid, gate_start_grid, gate_end_grid),
            start=1,
        ):
            if gate_start >= gate_end:
                continue
            candidates.append(
                Candidate(
                    group="d1",
                    run_id=f"D1S{idx:02d}",
                    overrides={
                        "model.lambda_high_plddt_l2": high_l2,
                        "model.lambda_low_plddt_l2": low_l2,
                        "model.plddt_gate_start": gate_start,
                        "model.plddt_gate_end": gate_end,
                    },
                )
            )

    return candidates


def metric_pattern(metric: str) -> re.Pattern[str]:
    if metric == "disp1to5":
        return re.compile(r"best-disp1to5-\\d+-([0-9]*\\.?[0-9]+)\\.ckpt$")
    return re.compile(r"best-flex-\\d+-([0-9]*\\.?[0-9]+)\\.ckpt$")


def extract_metric_value(ckpt_run_dir: Path, metric: str) -> float:
    pattern = metric_pattern(metric)
    for path in ckpt_run_dir.glob("*.ckpt"):
        match = pattern.match(path.name)
        if match:
            return float(match.group(1))
    raise RuntimeError(f"Could not find metric checkpoint in {ckpt_run_dir}")


def build_command(candidate: Candidate, args: argparse.Namespace) -> tuple[list[str], str]:
    merged = dict(BASELINE_OFF_OVERRIDES)
    merged.update(candidate.overrides)

    study_name = f"b1_d1_param_search/{candidate.group}/{candidate.run_id}_seed{args.seed}"
    cmd = [
        args.python,
        "train.py",
        f"seed={args.seed}",
        f"study_name={study_name}",
    ]
    cmd.extend(f"{k}={v}" for k, v in merged.items())
    cmd.extend(args.extra_override)
    return cmd, study_name


def newest_subdir(path: Path) -> Path:
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if not subdirs:
        raise RuntimeError(f"No run directories found under {path}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = parse_args()
    candidates = build_candidates(args.group)
    if args.max_runs > 0:
        candidates = candidates[: args.max_runs]

    if not candidates:
        raise RuntimeError("No candidates selected. Check --group and --max-runs settings.")

    best_by_group: dict[str, tuple[float, Candidate]] = {}

    for candidate in candidates:
        cmd, study_name = build_command(candidate, args)
        quoted = " ".join(shlex.quote(part) for part in cmd)
        print(f"[{candidate.run_id}] group={candidate.group} overrides={candidate.overrides}")
        print(f"  {quoted}")

        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)

        ckpt_root = Path("checkpoints") / study_name
        run_dir = newest_subdir(ckpt_root)
        value = extract_metric_value(run_dir, args.metric)
        print(f"  -> {args.metric}={value:.4f}")

        current_best = best_by_group.get(candidate.group)
        if current_best is None or value < current_best[0]:
            best_by_group[candidate.group] = (value, candidate)

    print("\n=== Best settings ===")
    if args.dry_run:
        print("Dry-run mode: no training executed, so no best metric available yet.")
        return

    for group in sorted(best_by_group):
        value, candidate = best_by_group[group]
        print(f"{group.upper()}: {args.metric}={value:.4f} with {candidate.run_id} overrides={candidate.overrides}")


if __name__ == "__main__":
    main()
