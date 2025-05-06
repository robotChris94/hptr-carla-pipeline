#!/usr/bin/env python3
"""
metrics.py
用法:
    python metrics.py --pred results/g2/pred.npy --gt data/gt.npy --out metrics.json

輸入:
    pred.npy : (N, 30, 2) 模型預測的連續座標 (x, y)
    gt.npy   : (N, 30, 2) 真實軌跡
輸出:
    JSON 檔含
        ADE  : 平均位移誤差
        FDE  : 終點位移誤差
        COL  : Collision Rate (0~1)
"""

import numpy as np
import json
import argparse
from pathlib import Path


def ade(pred: np.ndarray, gt: np.ndarray) -> float:
    return np.mean(np.linalg.norm(pred - gt, axis=-1))


def fde(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred[:, -1] - gt[:, -1], axis=-1)))


def collision_rate(pred: np.ndarray, gt: np.ndarray, thresh: float = 2.0) -> float:
    """
    collision: if any timestep can distance < thresh, then it decide to collision
    pred, gt shape = (N, 30, 2) ; only focus on ego-car
    """
    dist = np.linalg.norm(pred - gt, axis=-1)      # (N, 30)
    col = (dist < thresh).any(axis=1)
    return float(np.mean(col))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="path to pred.npy")
    ap.add_argument("--gt", required=True, help="path to gt.npy")
    ap.add_argument("--out", default="metrics.json", help="output json")
    args = ap.parse_args()

    pred = np.load(args.pred)        # shape = (N, 30, 2)
    gt = np.load(args.gt)

    results = {
        "ADE": float(ade(pred, gt)),
        "FDE": float(fde(pred, gt)),
        "COL": float(collision_rate(pred, gt))
    }

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"✅  metrics saved to {args.out}")


if __name__ == "__main__":
    main()
