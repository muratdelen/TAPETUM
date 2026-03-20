"""
Configuration file for RetinexTapetum.

This file keeps all experiment paths and hyperparameters in one place so that
training, validation, and test scripts use exactly the same setup.
"""

import os
import torch

# -----------------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------------
# LOL-v2 Real_captured dataset root.
DATA_ROOT = "../datasets/LoLv2/LOL-v2/Real_captured"

# Paired low-light / normal-light folders used during training and validation.
TRAIN_LOW_DIR = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")
VAL_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")
VAL_HIGH_DIR = os.path.join(DATA_ROOT, "Test", "Normal")
TEST_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")
TEST_HIGH_DIR = os.path.join(DATA_ROOT, "Test", "Normal")

# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------
# All experiment artifacts are written under this run directory.
RUN_ROOT = "../LoLv2/RetinexTapetum"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
RESULT_DIR = os.path.join(RUN_ROOT, "results", "Test")

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Data loading / optimization hyperparameters
# -----------------------------------------------------------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BATCH_SIZE = 8
NUM_WORKERS = 2
CROP_SIZE = 256
EPOCHS = 120
LR = 2e-4
GRAD_CLIP_NORM = 5.0

# -----------------------------------------------------------------------------
# Tapetum illumination enhancement hyperparameters
# -----------------------------------------------------------------------------
# Reference-aligned bounded lambda:
#   lambda = lambda_max * sigmoid(lambda_param)
# This avoids overly aggressive enhancement growth and gives smoother training.
LAMBDA_MAX = 1.68
LAMBDA_INIT = 0.0

# -----------------------------------------------------------------------------
# Main enhancement loss weights
# -----------------------------------------------------------------------------
W_L1 = 1.0
W_SSIM = 0.4
W_COLOR = 0.08
W_ATTN = 0.012

# -----------------------------------------------------------------------------
# Retinex decomposition loss weights
# -----------------------------------------------------------------------------
W_RECON_LOW = 1.0
W_RECON_HIGH = 1.0
W_REFLECT = 0.08
W_SMOOTH_LOW = 0.12
W_SMOOTH_HIGH = 0.12
# Extra smoothness constraint on enhanced illumination L_t.
W_SMOOTH_ENH = 0.10

# -----------------------------------------------------------------------------
# Training control
# -----------------------------------------------------------------------------
PATIENCE = 50
SEED = 42
BASE_CHANNELS = 32
