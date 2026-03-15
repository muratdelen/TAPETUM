"""Common utilities used by train and test scripts."""

import csv
import math
import random
import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed=42):
    """Seed Python, NumPy and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calc_psnr(pred, target):
    """Compute PSNR assuming images are in the [0, 1] range."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def write_history_csv(history, csv_path):
    """Export epoch history to CSV for easy comparison across models."""
    if not history:
        return

    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
