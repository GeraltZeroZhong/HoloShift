import argparse
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.datamodule import EvoPointDataModule
from evopoint_da.models.module import EvoPointLitModule


def get_args():
    p = argparse.ArgumentParser(description="Compute conformal q-hat from calibration set.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data/processed_graphs")
    p.add_argument("--data_cfg", default="configs/data/protein_displacement.yaml")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--output", default="artifacts/conformal_stats.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--calib_batch_size", type=int, default=None)
    p.add_argument("--split_seed", type=int, default=None)
    p.add_argument("--fallback_num_features", type=int, default=None)
    return p.parse_args()


def main():
    args = get_args()
    cfg = OmegaConf.load(args.data_cfg) if os.path.exists(args.data_cfg) else OmegaConf.create({})
    dm_kwargs = {
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "calib_batch_size": args.calib_batch_size if args.calib_batch_size is not None else cfg.get("calib_batch_size", 1),
        "split_seed": args.split_seed if args.split_seed is not None else cfg.get("split_seed", 42),
        "fallback_num_features": (
            args.fallback_num_features if args.fallback_num_features is not None else cfg.get("fallback_num_features", 144)
        ),
        "split_ranges": cfg.get("split_ranges", None),
    }
    dm = EvoPointDataModule(**dm_kwargs)
    dm.setup("fit")
    loader = dm.calib_dataloader()

    model = EvoPointLitModule.load_from_checkpoint(args.ckpt, map_location=args.device)
    model.eval().to(args.device)

    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            pred = model.predict_displacement(batch)
            s = torch.norm(pred - batch.y, dim=-1)
            scores.extend(s.detach().cpu().numpy().tolist())

    scores = np.array(scores, dtype=np.float32)
    n = len(scores)
    q = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - args.alpha)) / n), method="higher") if n > 0 else float("nan")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload = {"alpha": args.alpha, "num_calibration_nodes": int(n), "qhat": float(q)}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
