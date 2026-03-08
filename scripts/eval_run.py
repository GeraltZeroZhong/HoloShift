import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.datamodule import EvoPointDataModule
from evopoint_da.models.module import EvoPointLitModule


def get_args():
    p = argparse.ArgumentParser(description="Compute conformal q-hat from calibration set.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="data/processed_graphs")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--output", default="artifacts/conformal_stats.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()
    dm = EvoPointDataModule(data_dir=args.data_dir, batch_size=1, num_workers=0)
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
