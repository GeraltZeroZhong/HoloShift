import argparse
import glob
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def get_args():
    p = argparse.ArgumentParser(description="Baseline sanity-check using flattened node features.")
    p.add_argument("--data_dir", default="data/processed_graphs")
    return p.parse_args()


def main():
    args = get_args()
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))
    if not files:
        raise ValueError("No .pt graph files found")

    x_all, y_all = [], []
    for f in files:
        d = torch.load(f, weights_only=False)
        x_all.append(d["x"].numpy())
        y_all.append(torch.norm(d["y_delta"], dim=1).numpy())

    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    n = len(y)
    split = int(n * 0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)

    print(f"Samples: train={len(y_train)} test={len(y_test)}")
    print(f"MAE: {mean_absolute_error(y_test, pred):.6f}")
    print(f"R2 : {r2_score(y_test, pred):.6f}")


if __name__ == "__main__":
    main()
