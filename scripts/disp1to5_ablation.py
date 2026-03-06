#!/usr/bin/env python3
"""Plan and analyze targeted ablations for disp_1to5 optimization.

This script keeps the run count low while making parameter contribution checks explicit.

Usage examples:
  python scripts/disp1to5_ablation.py plan
  python scripts/disp1to5_ablation.py plan --replicate-top 2 --replicate-seeds 42,1337,2024
  python scripts/disp1to5_ablation.py analyze --results artifacts/disp1to5_ablation_results.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class RunSpec:
    run_id: str
    phase: str
    disp_focus_weight: float
    disp_outside_focus_weight: float
    mse_hard_beta: float
    lambda_cos: float
    lambda_mag: float
    notes: str


def default_matrix() -> List[RunSpec]:
    """Staged matrix to isolate contributions while avoiding duplicate/low-value runs."""
    return [
        RunSpec("A0", "focus", 1.00, 1.00, 3.0, 1.0, 1.0, "control; no focus weighting"),
        RunSpec("A1", "focus", 1.25, 0.95, 3.0, 1.0, 1.0, "mild focus"),
        RunSpec("A2", "focus", 1.50, 0.90, 3.0, 1.0, 1.0, "current focus default"),
        RunSpec("A3", "focus", 2.00, 0.80, 3.0, 1.0, 1.0, "aggressive focus"),
        RunSpec("B0", "hard",  1.50, 0.90, 0.0, 1.0, 1.0, "remove hard-range boost"),
        RunSpec("B1", "hard",  1.50, 0.90, 1.5, 1.0, 1.0, "moderate hard-range boost"),
        RunSpec("B3", "hard",  1.50, 0.90, 4.5, 1.0, 1.0, "stronger hard-range boost"),
        RunSpec("C1", "aux",   1.50, 0.90, 3.0, 0.5, 1.0, "lower cosine weight"),
        RunSpec("C2", "aux",   1.50, 0.90, 3.0, 1.0, 0.5, "lower magnitude weight"),
        RunSpec("C3", "aux",   1.50, 0.90, 3.0, 0.5, 0.5, "lower both aux terms"),
        RunSpec("D1", "inter", 1.50, 0.90, 0.0, 0.5, 0.5, "check if focus alone drives gains"),
    ]


def hydra_cmd(spec: RunSpec, seed: int = 42) -> str:
    return (
        "python train.py "
        f"seed={seed} "
        f"study_name=disp1to5_ablation/{spec.run_id}_seed{seed} "
        f"model.disp_focus_weight={spec.disp_focus_weight} "
        f"model.disp_outside_focus_weight={spec.disp_outside_focus_weight} "
        f"model.mse_hard_beta={spec.mse_hard_beta} "
        f"model.lambda_cos={spec.lambda_cos} "
        f"model.lambda_mag={spec.lambda_mag}"
    )


def print_plan(replicate_top: int, replicate_seeds: List[int]) -> None:
    runs = default_matrix()
    print(f"# Phase-1 matrix ({len(runs)} runs)")
    for r in runs:
        print(f"[{r.run_id}] {r.notes}")
        print(f"  {hydra_cmd(r)}")

    if replicate_top > 0 and replicate_seeds:
        print("\n# Phase-2 replication (trustworthiness)")
        print(
            "After phase-1, pick top runs by test/disp_1to5_mse passing safety gates, "
            f"then re-run top {replicate_top} across seeds: {replicate_seeds}."
        )
        for i in range(1, replicate_top + 1):
            print(f"  Top-{i} template:")
            for s in replicate_seeds:
                print(f"    {hydra_cmd(RunSpec('TOP', 'rep', 1.5, 0.9, 3.0, 1.0, 1.0, ''), seed=s)}")

    print("\n# Recommended acceptance gates")
    print("1) Improve test/disp_1to5_mse by >= 5% vs control (A0).")
    print("2) test/loss_mse degradation <= 10% vs control.")
    print("3) test/disp_0to0p5_mse and test/disp_0p5to1_mse each degradation <= 15% vs control.")


def _float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Missing/invalid numeric column '{key}'") from exc


def analyze(results_path: Path) -> None:
    if not results_path.exists():
        raise FileNotFoundError(results_path)

    with results_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = [
        "run_id",
        "test_disp_1to5_mse",
        "test_loss_mse",
        "test_disp_0to0p5_mse",
        "test_disp_0p5to1_mse",
        "disp_focus_weight",
        "disp_outside_focus_weight",
        "mse_hard_beta",
        "lambda_cos",
        "lambda_mag",
    ]
    missing = [c for c in required if not rows or c not in rows[0]]
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")

    # Rank by primary objective.
    ranked = sorted(rows, key=lambda r: _float(r, "test_disp_1to5_mse"))
    print("# Top runs by test_disp_1to5_mse")
    for r in ranked[:5]:
        print(
            f"{r['run_id']}: disp_1to5={_float(r, 'test_disp_1to5_mse'):.6f}, "
            f"loss_mse={_float(r, 'test_loss_mse'):.6f}, "
            f"disp_0to0p5={_float(r, 'test_disp_0to0p5_mse'):.6f}, "
            f"disp_0p5to1={_float(r, 'test_disp_0p5to1_mse'):.6f}"
        )

    # Simple marginal-effect estimates (mean at level - global mean).
    objective = [_float(r, "test_disp_1to5_mse") for r in rows]
    gmean = sum(objective) / max(len(objective), 1)

    params = ["disp_focus_weight", "disp_outside_focus_weight", "mse_hard_beta", "lambda_cos", "lambda_mag"]
    print("\n# Parameter contribution check (marginal mean delta on disp_1to5_mse; lower is better)")
    print(f"Global mean disp_1to5_mse: {gmean:.6f}")
    for p in params:
        buckets: Dict[str, List[float]] = {}
        for r in rows:
            buckets.setdefault(r[p], []).append(_float(r, "test_disp_1to5_mse"))
        print(f"- {p}")
        for level, vals in sorted(buckets.items(), key=lambda kv: float(kv[0])):
            m = sum(vals) / len(vals)
            print(f"    level={level:>6}: mean={m:.6f} (delta={m - gmean:+.6f}, n={len(vals)})")

    print("\nInterpretation: more negative delta indicates stronger contribution to improving disp_1to5.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plan/analyze disp_1to5 ablations")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="print staged run matrix and commands")
    p_plan.add_argument("--replicate-top", type=int, default=2)
    p_plan.add_argument("--replicate-seeds", type=str, default="42,1337,2024")

    p_analyze = sub.add_parser("analyze", help="analyze completed runs from CSV")
    p_analyze.add_argument("--results", type=Path, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "plan":
        seeds = [int(s.strip()) for s in args.replicate_seeds.split(",") if s.strip()]
        print_plan(args.replicate_top, seeds)
    elif args.cmd == "analyze":
        analyze(args.results)
    else:
        raise RuntimeError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
