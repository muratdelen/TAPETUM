"""Utility helpers shared by train and test scripts."""

import csv
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed=42):
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calc_psnr(pred, target):
    """Compute PSNR assuming image tensors are in [0,1]."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def save_history_csv(history, csv_path):
    """Save epoch history to CSV for later comparison across models."""
    if not history:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = []
    for item in history:
        row = {
            "epoch": item["epoch"],
            "lambda": item["lambda"],
            "gate_r": item["channel_gate"][0],
            "gate_g": item["channel_gate"][1],
            "gate_b": item["channel_gate"][2],
        }
        for prefix in ("train", "val"):
            for k, v in item[prefix].items():
                row[f"{prefix}_{k}"] = v
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
