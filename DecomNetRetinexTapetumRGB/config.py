import os
import torch

# ===== Paths =====
DATA_ROOT = "../datasets/LoLv2/LOL-v2/Real_captured"

TRAIN_LOW_DIR = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")
VAL_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")
VAL_HIGH_DIR = os.path.join(DATA_ROOT, "Test", "Normal")
TEST_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")
TEST_HIGH_DIR = os.path.join(DATA_ROOT, "Test", "Normal")

RUN_ROOT = "../LoLv2/DecomNetRetinexTapetumRGB"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
RESULT_DIR = os.path.join(RUN_ROOT, "results", "Test")

# ===== Device =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Hyperparameters =====
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BATCH_SIZE = 8
NUM_WORKERS = 2
CROP_SIZE = 256
EPOCHS = 120
LR = 2e-4

CHANNEL_GATE_SCALE = 0.25
LAMBDA_MAX = 1.68

# enhancement loss weights
W_L1 = 1.0
W_SSIM = 0.4
W_COLOR = 0.08
W_ATTN = 0.012
W_GATE = 0.003

# decomposition loss weights
W_RECON_LOW = 1.0
W_RECON_HIGH = 1.0
W_REFLECT = 0.08
W_SMOOTH_LOW = 0.12
W_SMOOTH_HIGH = 0.12
W_SMOOTH_ENH = 0.10

# early stopping
PATIENCE = 50

SEED = 42
BASE_CHANNELS = 32
# sigmoid(0)=0.5 -> lam starts at 0.84 when lambda_max=1.68
LAMBDA_INIT = 0.0
GRAD_CLIP_NORM = 5.0
