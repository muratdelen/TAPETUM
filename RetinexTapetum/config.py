"""
RetinexTapetum configuration
----------------------------
This file stores all path, optimization and logging parameters in one place.
The goal is to keep this model aligned with the same experimental standard used
for the other TAPETUM variants so that training logs and final results can be
compared fairly.
"""

import os
import torch

# ============================================================
# Dataset paths
# ============================================================
DATA_ROOT = "../datasets/LoLv2/LOL-v2/Real_captured"

TRAIN_LOW_DIR = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")
VAL_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")
VAL_HIGH_DIR = os.path.join(DATA_ROOT, "Test", "Normal")
TEST_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")

# ============================================================
# Run directories
# ============================================================
RUN_ROOT = "../LoLv2/retinex-tapetum"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
RESULT_DIR = os.path.join(RUN_ROOT, "results", "Test")
HISTORY_CSV = os.path.join(RUN_ROOT, "history.csv")

# ============================================================
# System
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ============================================================
# Data settings
# ============================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BATCH_SIZE = 8
NUM_WORKERS = 2
CROP_SIZE = 256

# ============================================================
# Optimization
# ============================================================
EPOCHS = 120
LR = 2e-4
PATIENCE = 50
GRAD_CLIP_NORM = 5.0

# ============================================================
# Model settings
# ============================================================
BASE_CHANNELS = 32
LAMBDA_INIT = 0.0
LAMBDA_MAX = 1.68

# Gaussian Retinex decomposition hyperparameters
GAUSSIAN_KERNEL_SIZE = 21
GAUSSIAN_SIGMA = 5.0

# ============================================================
# Loss weights
# ============================================================
W_L1 = 1.0
W_SSIM = 0.4
W_COLOR = 0.08
W_ATTN = 0.012
W_SMOOTH_ENH = 0.10
