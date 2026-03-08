import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.datamodule import EvoPointDataModule
from evopoint_da.models.module import EvoPointLitModule


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Point-wise regression analysis between per-residue pLDDT and Euclidean coordinate error "
            "for model prediction and zero-displacement baseline."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint (.ckpt).")
    parser.add_argument("--data_dir", default="data/processed_graphs", help="Directory with processed graph .pt files.")
    parser.add_argument("--split", choices=["test", "calib", "val", "train"], default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bins", type=int, default=10, help="Number of equally spaced pLDDT bins in [0, 100].")
    parser.add_argument("--output_json", default="artifacts/pointwise_plddt_regression.json")
    parser.add_argument("--output_csv", default="artifacts/pointwise_plddt_samples.csv")
    parser.add_argument("--output_plot", default="artifacts/pointwise_plddt_regression.png")
    return parser.parse_args()


def _as_float(v: torch.Tensor) -> float:
    return float(v.detach().cpu().item())


def safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_std = torch.sqrt((x_centered**2).mean())
    y_std = torch.sqrt((y_centered**2).mean())
    if _as_float(x_std) <= 0.0 or _as_float(y_std) <= 0.0:
        return float("nan")
    cov = (x_centered * y_centered).mean()
    return _as_float(cov / (x_std * y_std))


def rankdata(a: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(a, stable=True)
    ranks = torch.empty_like(a, dtype=torch.float64)
    ranks[order] = torch.arange(a.numel(), device=a.device, dtype=torch.float64)

    sorted_vals = a[order]
    i = 0
    n = a.numel()
    while i < n:
        j = i + 1
        while j < n and _as_float(sorted_vals[j] - sorted_vals[i]) == 0.0:
            j += 1
        avg_rank = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    return safe_corrcoef(rankdata(x), rankdata(y))


def linear_fit(x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    if x.numel() < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}

    x_mean = x.mean()
    y_mean = y.mean()
    x_centered = x - x_mean
    y_centered = y - y_mean
    denom = (x_centered**2).sum()
    if _as_float(denom) <= 0.0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}

    slope = (x_centered * y_centered).sum() / denom
    intercept = y_mean - slope * x_mean
    y_hat = slope * x + intercept

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = float("nan") if _as_float(ss_tot) <= 0.0 else _as_float(1.0 - ss_res / ss_tot)

    return {
        "slope": _as_float(slope),
        "intercept": _as_float(intercept),
        "r2": r2,
    }


def summarize_by_bins(plddt: torch.Tensor, values: torch.Tensor, bins: int) -> List[Dict[str, float]]:
    step = 100.0 / bins
    out = []
    for i in range(bins):
        lo = i * step
        hi = (i + 1) * step
        if i == bins - 1:
            mask = (plddt >= lo) & (plddt <= hi)
            right_bracket = "]"
        else:
            mask = (plddt >= lo) & (plddt < hi)
            right_bracket = ")"

        if mask.any():
            v = values[mask]
            out.append(
                {
                    "bin": f"[{lo:.1f}, {hi:.1f}{right_bracket}",
                    "count": int(mask.sum().item()),
                    "mean": _as_float(v.mean()),
                    "median": _as_float(v.median()),
                    "std": _as_float(v.std(unbiased=False)),
                }
            )
        else:
            out.append(
                {
                    "bin": f"[{lo:.1f}, {hi:.1f}{right_bracket}",
                    "count": 0,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "std": float("nan"),
                }
            )
    return out


def save_regression_plot(
    plddt: torch.Tensor,
    pred_err: torch.Tensor,
    zero_err: torch.Tensor,
    pred_fit: Dict[str, float],
    zero_fit: Dict[str, float],
    output_path: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed; skipping regression plot generation.", file=sys.stderr)
        return False

    plt.figure(figsize=(10, 6))

    plddt_np = plddt.cpu().numpy()
    pred_err_np = pred_err.cpu().numpy()
    zero_err_np = zero_err.cpu().numpy()

    plt.scatter(plddt_np, pred_err_np, s=8, alpha=0.25, label="Predicted error", color="#1f77b4")
    plt.scatter(plddt_np, zero_err_np, s=8, alpha=0.25, label="Zero-displacement error", color="#ff7f0e")

    x_line = torch.linspace(0.0, 100.0, 200, dtype=torch.float64)
    if not (math.isnan(pred_fit["slope"]) or math.isnan(pred_fit["intercept"])):
        y_pred_line = pred_fit["slope"] * x_line + pred_fit["intercept"]
        plt.plot(x_line.numpy(), y_pred_line.numpy(), color="#1f77b4", linewidth=2.0, label="Predicted regression")

    if not (math.isnan(zero_fit["slope"]) or math.isnan(zero_fit["intercept"])):
        y_zero_line = zero_fit["slope"] * x_line + zero_fit["intercept"]
        plt.plot(
            x_line.numpy(),
            y_zero_line.numpy(),
            color="#ff7f0e",
            linewidth=2.0,
            linestyle="--",
            label="Zero-displacement regression",
        )

    plt.xlabel("pLDDT")
    plt.ylabel("Per-residue Euclidean error")
    plt.title("Point-wise pLDDT vs. coordinate error")
    plt.xlim(0, 100)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()

    plot_dir = os.path.dirname(output_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def main():
    args = parse_args()

    dm = EvoPointDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup("fit" if args.split in {"train", "val", "calib"} else "test")

    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "val":
        loader = dm.val_dataloader()
    elif args.split == "calib":
        loader = dm.calib_dataloader()
    else:
        loader = dm.test_dataloader()

    model = EvoPointLitModule.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval().to(args.device)

    plddt_all = []
    pred_err_all = []
    zero_err_all = []

    with torch.no_grad():
        for batch in loader:
            if not hasattr(batch, "plddt") or batch.plddt is None:
                continue

            batch = batch.to(args.device)
            pred_disp = model.predict_displacement(batch)
            pred_error = torch.norm(pred_disp - batch.y, dim=-1)
            zero_error = torch.norm(batch.y, dim=-1)

            plddt = batch.plddt
            if plddt.dim() > 1:
                plddt = plddt.squeeze(-1)

            plddt_all.append(plddt.detach().cpu().to(torch.float64))
            pred_err_all.append(pred_error.detach().cpu().to(torch.float64))
            zero_err_all.append(zero_error.detach().cpu().to(torch.float64))

    if not plddt_all:
        raise RuntimeError("No pLDDT values found in selected split. Cannot run regression analysis.")

    plddt = torch.cat(plddt_all)
    pred_err = torch.cat(pred_err_all)
    zero_err = torch.cat(zero_err_all)

    pred_fit = linear_fit(plddt, pred_err)
    zero_fit = linear_fit(plddt, zero_err)

    payload = {
        "checkpoint": args.ckpt,
        "data_dir": args.data_dir,
        "split": args.split,
        "num_points": int(plddt.numel()),
        "predicted_vs_plddt": {
            "pearson": safe_corrcoef(plddt, pred_err),
            "spearman": spearman_corr(plddt, pred_err),
            **pred_fit,
        },
        "zero_displacement_vs_plddt": {
            "pearson": safe_corrcoef(plddt, zero_err),
            "spearman": spearman_corr(plddt, zero_err),
            **zero_fit,
        },
        "predicted_error_by_plddt_bins": summarize_by_bins(plddt, pred_err, args.bins),
        "zero_error_by_plddt_bins": summarize_by_bins(plddt, zero_err, args.bins),
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    csv_dir = os.path.dirname(args.output_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plddt", "predicted_error", "zero_displacement_error"])
        for p, pe, ze in zip(plddt.tolist(), pred_err.tolist(), zero_err.tolist()):
            writer.writerow([p, pe, ze])

    plot_saved = save_regression_plot(plddt, pred_err, zero_err, pred_fit, zero_fit, args.output_plot)

    print(json.dumps(payload, indent=2, allow_nan=True))
    print(f"Saved pointwise samples to: {args.output_csv}")
    if plot_saved:
        print(f"Saved regression plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
