#!/usr/bin/env python3
"""Plan, extract and analyze targeted ablations for disp_1to5 optimization.

Usage examples:
  python scripts/disp1to5_ablation.py plan
  python scripts/disp1to5_ablation.py extract
  python scripts/disp1to5_ablation.py analyze --results artifacts/disp1to5_ablation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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
        RunSpec("B0", "hard", 1.50, 0.90, 0.0, 1.0, 1.0, "remove hard-range boost"),
        RunSpec("B1", "hard", 1.50, 0.90, 1.5, 1.0, 1.0, "moderate hard-range boost"),
        RunSpec("B3", "hard", 1.50, 0.90, 4.5, 1.0, 1.0, "stronger hard-range boost"),
        RunSpec("C1", "aux", 1.50, 0.90, 3.0, 0.5, 1.0, "lower cosine weight"),
        RunSpec("C2", "aux", 1.50, 0.90, 3.0, 1.0, 0.5, "lower magnitude weight"),
        RunSpec("C3", "aux", 1.50, 0.90, 3.0, 0.5, 0.5, "lower both aux terms"),
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


RESULTS_COLUMNS = [
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

REQUIRED_METRIC_COLUMNS = ["test_disp_1to5_mse", "test_loss_mse"]
OPTIONAL_SAFETY_METRIC_COLUMNS = ["test_disp_0to0p5_mse", "test_disp_0p5to1_mse"]

RE_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"

METRIC_PATTERNS = {
    "test_disp_1to5_mse": re.compile(rf"test[\/_]disp_1to5_mse[^0-9eE+\-]*{RE_FLOAT}"),
    "test_loss_mse": re.compile(rf"test[\/_]loss_mse[^0-9eE+\-]*{RE_FLOAT}"),
    "test_disp_0to0p5_mse": re.compile(rf"test[\/_]disp_0to0p5_mse[^0-9eE+\-]*{RE_FLOAT}"),
    "test_disp_0p5to1_mse": re.compile(rf"test[\/_]disp_0p5to1_mse[^0-9eE+\-]*{RE_FLOAT}"),
}

PARAM_OVERRIDES = {
    "disp_focus_weight": re.compile(rf"(?:model\.)?disp_focus_weight\s*[=:]\s*{RE_FLOAT}"),
    "disp_outside_focus_weight": re.compile(rf"(?:model\.)?disp_outside_focus_weight\s*[=:]\s*{RE_FLOAT}"),
    "mse_hard_beta": re.compile(rf"(?:model\.)?mse_hard_beta\s*[=:]\s*{RE_FLOAT}"),
    "lambda_cos": re.compile(rf"(?:model\.)?lambda_cos\s*[=:]\s*{RE_FLOAT}"),
    "lambda_mag": re.compile(rf"(?:model\.)?lambda_mag\s*[=:]\s*{RE_FLOAT}"),
}


def _default_param_map() -> Dict[str, Dict[str, float]]:
    mapping: Dict[str, Dict[str, float]] = {}
    for spec in default_matrix():
        mapping[spec.run_id] = {
            "disp_focus_weight": spec.disp_focus_weight,
            "disp_outside_focus_weight": spec.disp_outside_focus_weight,
            "mse_hard_beta": spec.mse_hard_beta,
            "lambda_cos": spec.lambda_cos,
            "lambda_mag": spec.lambda_mag,
        }
    return mapping


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _find_last_value(text: str, pattern: re.Pattern[str]) -> Optional[float]:
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _collect_candidate_logs(search_roots: Iterable[Path]) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p in seen:
                continue
            if p.suffix.lower() in {".log", ".txt", ".out", ".err", ".json", ".yaml", ".yml", ".csv"}:
                seen.add(p)
                candidates.append(p)
                continue
            if "log" in p.name.lower():
                seen.add(p)
                candidates.append(p)
    return sorted(candidates, key=lambda x: (x.stat().st_mtime, str(x)))


def _iter_csv_key_aliases(base_keys: Iterable[str]) -> List[str]:
    aliases: List[str] = []
    for key in base_keys:
        aliases.append(key)
        aliases.append(f"{key}_epoch")
        aliases.append(f"{key}_step")
        aliases.append(f"{key}/dataloader_idx_0")
        aliases.append(f"{key}_dataloader_idx_0")
        aliases.append(f"{key}_epoch/dataloader_idx_0")
        aliases.append(f"{key}_epoch_dataloader_idx_0")
    # preserve insertion order while removing duplicates
    return list(dict.fromkeys(aliases))


def _extract_float_from_csv(path: Path, keys: Iterable[str]) -> Optional[float]:
    alias_keys = _iter_csv_key_aliases(keys)
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in alias_keys:
                    if key not in row:
                        continue
                    value = (row.get(key) or "").strip()
                    if not value:
                        continue
                    try:
                        last = float(value)
                    except ValueError:
                        continue
            return locals().get("last")
    except Exception:
        return None


def _collect_available_metric_keys(log_files: Iterable[Path]) -> List[str]:
    metricish: set[str] = set()
    for log_path in log_files:
        if log_path.suffix.lower() != ".csv":
            continue
        try:
            with log_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
        except Exception:
            continue

        for h in headers:
            if not h:
                continue
            low = h.lower()
            if any(tok in low for tok in ("test", "val", "disp", "mse", "loss")):
                metricish.add(h)
    return sorted(metricish)


def _extract_metrics_from_logs(log_files: Iterable[Path]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    csv_keys = {
        "test_disp_1to5_mse": ["test/disp_1to5_mse", "test_disp_1to5_mse"],
        "test_loss_mse": ["test/loss_mse", "test_loss_mse"],
        "test_disp_0to0p5_mse": ["test/disp_0to0p5_mse", "test_disp_0to0p5_mse"],
        "test_disp_0p5to1_mse": ["test/disp_0p5to1_mse", "test_disp_0p5to1_mse"],
    }

    for log_path in log_files:
        if log_path.suffix.lower() == ".csv":
            for key, aliases in csv_keys.items():
                value = _extract_float_from_csv(log_path, aliases)
                if value is not None:
                    metrics[key] = value
            continue

        text = _read_text(log_path)
        if not text:
            continue
        for key, pattern in METRIC_PATTERNS.items():
            value = _find_last_value(text, pattern)
            if value is not None:
                metrics[key] = value
    return metrics


def _extract_params_from_logs(log_files: Iterable[Path]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    csv_keys = {
        "disp_focus_weight": ["model.disp_focus_weight", "disp_focus_weight"],
        "disp_outside_focus_weight": ["model.disp_outside_focus_weight", "disp_outside_focus_weight"],
        "mse_hard_beta": ["model.mse_hard_beta", "mse_hard_beta"],
        "lambda_cos": ["model.lambda_cos", "lambda_cos"],
        "lambda_mag": ["model.lambda_mag", "lambda_mag"],
    }

    for log_path in log_files:
        if log_path.suffix.lower() == ".csv":
            for key, aliases in csv_keys.items():
                value = _extract_float_from_csv(log_path, aliases)
                if value is not None:
                    params[key] = value
            continue

        text = _read_text(log_path)
        if not text:
            continue
        for key, pattern in PARAM_OVERRIDES.items():
            value = _find_last_value(text, pattern)
            if value is not None:
                params[key] = value
    return params


def _parse_run_id_seed(run_dir_name: str) -> Tuple[str, Optional[int]]:
    m = re.fullmatch(r"(.+)_seed(\d+)", run_dir_name)
    if not m:
        return run_dir_name, None
    return m.group(1), int(m.group(2))


def _latest_timestamp_dir(run_dir: Path) -> Optional[Path]:
    ts_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
    if not ts_dirs:
        return None
    return sorted(ts_dirs, key=lambda p: p.name)[-1]


def _hydra_output_dir_from_timestamp(repo_root: Path, timestamp: str) -> Optional[Path]:
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})", timestamp)
    if not m:
        return None
    day = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    t = f"{m.group(4)}-{m.group(5)}-{m.group(6)}"
    candidate = repo_root / "outputs" / day / t
    if candidate.exists() and candidate.is_dir():
        return candidate
    return None


def _csv_logger_dirs(repo_root: Path, limit: int = 5) -> List[Path]:
    """Return latest CSVLogger version dirs such as logs/holoshift/version_*/metrics.csv."""
    base = repo_root / "logs" / "holoshift"
    if not base.exists() or not base.is_dir():
        return []

    version_dirs = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("version_")]
    valid_dirs = [p for p in sorted(version_dirs, key=lambda x: x.name, reverse=True) if (p / "metrics.csv").exists()]
    return valid_dirs[:limit]


def collect_rows_from_logs(
    checkpoints_root: Path, strict: bool = False
) -> Tuple[List[Dict[str, float | str]], List[str]]:
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints root does not exist: {checkpoints_root}")

    repo_root = Path(__file__).resolve().parents[1]
    defaults = _default_param_map()
    rows: List[Dict[str, float | str]] = []
    warnings: List[str] = []
    run_dirs = sorted([p for p in checkpoints_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise RuntimeError(
            f"No run directories found under {checkpoints_root} (resolved: {checkpoints_root.resolve()}). "
            "Expected layout: <checkpoints_root>/<RUN_ID>_seed<SEED>/<timestamp>/."
        )

    for run_dir in run_dirs:
        run_id, _seed = _parse_run_id_seed(run_dir.name)
        ts_dir = _latest_timestamp_dir(run_dir)
        if ts_dir is None:
            continue

        search_roots: List[Path] = [ts_dir]
        hydra_dir = _hydra_output_dir_from_timestamp(repo_root, ts_dir.name)
        if hydra_dir is not None:
            search_roots.extend([hydra_dir, hydra_dir / "logs"])

        csv_logger_dirs = _csv_logger_dirs(repo_root)
        if csv_logger_dirs:
            search_roots.extend(csv_logger_dirs)

        log_files = _collect_candidate_logs(search_roots)
        metrics = _extract_metrics_from_logs(log_files)
        params = dict(defaults.get(run_id, {}))
        params.update(_extract_params_from_logs(log_files))

        row: Dict[str, float | str] = {"run_id": run_id}
        row.update(metrics)
        row.update(params)

        missing_required_metrics = [c for c in REQUIRED_METRIC_COLUMNS if c not in row]
        missing_optional_metrics = [c for c in OPTIONAL_SAFETY_METRIC_COLUMNS if c not in row]
        missing_other_columns = [c for c in RESULTS_COLUMNS if c not in row and c not in missing_optional_metrics]

        if missing_required_metrics or missing_other_columns:
            discovered = _collect_available_metric_keys(log_files)
            missing_all = missing_required_metrics + [c for c in missing_other_columns if c not in missing_required_metrics]
            message = (
                f"Run {run_dir.name} is missing required columns: {missing_all}. "
                f"Scanned roots: {', '.join(str(p) for p in search_roots)}"
            )
            if discovered:
                message += f". Available metric-like CSV keys include: {discovered[:10]}"
            else:
                message += ". No metric-like CSV headers were detected; this run likely has no test logs yet."
            if csv_logger_dirs:
                message += f" Checked CSVLogger dirs: {', '.join(str(p) for p in csv_logger_dirs)}"

            if strict:
                raise ValueError(message)
            warnings.append(message)
            continue

        for c in missing_optional_metrics:
            row[c] = float("nan")
        if missing_optional_metrics:
            warnings.append(
                f"Run {run_dir.name} is missing optional safety metrics {missing_optional_metrics}; filled with NaN."
            )

        rows.append(row)

    return rows, warnings


def write_results_csv(rows: List[Dict[str, float | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def extract_results(checkpoints_root: Path, output: Path, strict: bool = False) -> None:
    rows, warnings = collect_rows_from_logs(checkpoints_root, strict=strict)
    if not rows:
        detail = f" First incomplete run: {warnings[0]}" if warnings else ""
        raise RuntimeError(f"No complete runs found under {checkpoints_root}.{detail}")

    write_results_csv(rows, output)
    print(f"Wrote {len(rows)} rows to {output}")
    if warnings:
        print(f"\nNotes for {len(warnings)} runs with missing fields:")
        for w in warnings:
            print(f"- {w}")


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

    missing = [c for c in RESULTS_COLUMNS if not rows or c not in rows[0]]
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")

    ranked = sorted(rows, key=lambda r: _float(r, "test_disp_1to5_mse"))
    baseline_row = next((r for r in rows if r["run_id"] == "A0"), ranked[0])
    b_disp = _float(baseline_row, "test_disp_1to5_mse")
    b_loss = _float(baseline_row, "test_loss_mse")
    b_00 = _float(baseline_row, "test_disp_0to0p5_mse")
    b_05 = _float(baseline_row, "test_disp_0p5to1_mse")

    print("# Ranked ablation results (lower is better)")
    print(
        "run_id | disp_1to5 | ΔvsA0 | loss_mse | ΔvsA0 | 0to0p5 | ΔvsA0 | 0p5to1 | ΔvsA0 | gates"
    )
    print("-" * 112)
    for r in ranked:
        disp = _float(r, "test_disp_1to5_mse")
        loss = _float(r, "test_loss_mse")
        d00 = _float(r, "test_disp_0to0p5_mse")
        d05 = _float(r, "test_disp_0p5to1_mse")

        disp_gain = (b_disp - disp) / max(b_disp, 1e-12)
        loss_deg = (loss - b_loss) / max(b_loss, 1e-12)
        d00_deg = (d00 - b_00) / max(b_00, 1e-12) if math.isfinite(d00) and math.isfinite(b_00) else float("nan")
        d05_deg = (d05 - b_05) / max(b_05, 1e-12) if math.isfinite(d05) and math.isfinite(b_05) else float("nan")
        if r["run_id"] == baseline_row["run_id"]:
            gate_text = "BASELINE"
        else:
            if not (math.isfinite(d00_deg) and math.isfinite(d05_deg)):
                gate_text = "N/A"
            else:
                gates = (
                    (disp_gain >= 0.05),
                    (loss_deg <= 0.10),
                    (d00_deg <= 0.15),
                    (d05_deg <= 0.15),
                )
                gate_text = "PASS" if all(gates) else "FAIL"

        print(
            f"{r['run_id']:>5} | {disp:>9.6f} | {disp_gain:+6.2%} | {loss:>8.6f} | {loss_deg:+6.2%} | "
            f"{d00:>7.6f} | {d00_deg:+6.2%} | {d05:>7.6f} | {d05_deg:+6.2%} | {gate_text}"
        )

    print(
        f"\nBaseline for gate checks: run {baseline_row['run_id']} "
        f"(disp_1to5={b_disp:.6f}, loss_mse={b_loss:.6f})."
    )

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
    p = argparse.ArgumentParser(description="Plan/extract/analyze disp_1to5 ablations")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="print staged run matrix and commands")
    p_plan.add_argument("--replicate-top", type=int, default=2)
    p_plan.add_argument("--replicate-seeds", type=str, default="42,1337,2024")

    p_extract = sub.add_parser("extract", help="extract run metrics from logs into results CSV")
    p_extract.add_argument(
        "--checkpoints-root",
        type=Path,
        default=Path("checkpoints/disp1to5_ablation"),
        help="Root directory containing <RUN_ID>_seed<SEED>/<timestamp>/ subdirectories",
    )
    p_extract.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/disp1to5_ablation_results.csv"),
        help="Destination CSV path",
    )
    p_extract.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a run is incomplete instead of skipping it with warnings",
    )

    p_analyze = sub.add_parser("analyze", help="analyze completed runs from CSV")
    p_analyze.add_argument("--results", type=Path, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "plan":
        seeds = [int(s.strip()) for s in args.replicate_seeds.split(",") if s.strip()]
        print_plan(args.replicate_top, seeds)
    elif args.cmd == "extract":
        extract_results(args.checkpoints_root, args.output, strict=args.strict)
    elif args.cmd == "analyze":
        analyze(args.results)
    else:
        raise RuntimeError(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    main()
