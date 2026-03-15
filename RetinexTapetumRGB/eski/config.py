import os

# ===== Paths =====
DATA_ROOT = "/content/ADF-LLIE/datasets/LoLv2/LOL-v2/Real_captured"

TRAIN_LOW_DIR  = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")
VAL_LOW_DIR    = os.path.join(DATA_ROOT, "Test", "Low")
VAL_HIGH_DIR   = os.path.join(DATA_ROOT, "Test", "Normal")

RUN_ROOT = "/content/ADF-LLIE/LoLv2/RetinexTapetumRGB"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
RESULT_DIR = os.path.join(RUN_ROOT, "results")
LOG_CSV_PATH = os.path.join(RUN_ROOT, "history.csv")

# ===== Hyperparameters =====
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 0
CROP_SIZE = 256
EPOCHS = 120
LR = 2e-4

CHANNEL_GATE_SCALE = 0.25
BASE_CHANNELS = 32
LAMBDA_INIT = 1.0

# ===== Loss weights =====
W_L1 = 1.0
W_SSIM = 0.5
W_COLOR = 0.1
W_ATTN = 0.01
W_GATE = 0.002