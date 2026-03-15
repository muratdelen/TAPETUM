"""
RetinexTapetumRGB configuration file.

This file centralizes all experiment settings so that:
1. training and test scripts read the same paths,
2. all models in the project can be compared under the same protocol,
3. hyperparameter changes are visible in one place.
"""

import os
import torch

# ============================================================
# DATA PATHS
# ============================================================
# LOL-v2 Real_captured dataset root.
DATA_ROOT = "../datasets/LoLv2/LOL-v2/Real_captured"

# Paired training set.
TRAIN_LOW_DIR = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")

# Paired validation / test set.
VAL_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")
VAL_HIGH_DIR = os.path.join(DATA_ROOT, "Test", "Normal")

# Test script uses the low-light images from the official test split.
TEST_LOW_DIR = VAL_LOW_DIR

# ============================================================
# RUN OUTPUT PATHS
# ============================================================
RUN_ROOT = "../LoLv2/RetinexTapetumRGB"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
RESULT_DIR = os.path.join(RUN_ROOT, "results", "Test")
LOG_CSV_PATH = os.path.join(RUN_ROOT, "history.csv")

# Create main output folders proactively.
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# RUNTIME
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# DATA / TRAINING HYPERPARAMETERS
# ============================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 2
CROP_SIZE = 256
EPOCHS = 120
LR = 2e-4
PATIENCE = 50
GRAD_CLIP_NORM = 1.0

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================
BASE_CHANNELS = 32
CHANNEL_GATE_SCALE = 0.25

# Lambda is optimized as a free scalar parameter, but in forward pass it is
# constrained with a sigmoid. Therefore lambda_init is the raw parameter value.
LAMBDA_INIT = 0.0
LAMBDA_MAX = 1.68

# Gaussian Retinex decomposition parameters.
RETINEX_KERNEL_SIZE = 21
RETINEX_SIGMA = 5.0

# ============================================================
# LOSS WEIGHTS
# ============================================================
W_L1 = 1.0
W_SSIM = 0.4
W_COLOR = 0.08
W_ATTN = 0.012
W_GATE = 0.003
W_SMOOTH_ENH = 0.10
