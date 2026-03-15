import os
import torch

DATA_ROOT = "../datasets/LoLv2/LOL-v2/Real_captured"

TRAIN_LOW_DIR  = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")
VAL_LOW_DIR    = os.path.join(DATA_ROOT, "Test", "Low")
VAL_HIGH_DIR   = os.path.join(DATA_ROOT, "Test", "Normal")
TEST_LOW_DIR   = os.path.join(DATA_ROOT, "Test", "Low")
TEST_HIGH_DIR  = os.path.join(DATA_ROOT, "Test", "Normal")

RUN_ROOT = "../LoLv2/DecomNetRetinexTapetum"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
RESULT_DIR = os.path.join(RUN_ROOT, "results", "Test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BATCH_SIZE = 8
NUM_WORKERS = 2
CROP_SIZE = 256
EPOCHS = 120
LR = 2e-4

LAMBDA_MAX = 1.60

W_L1 = 1.0
W_SSIM = 0.4
W_COLOR = 0.08
W_ATTN = 0.02

W_RECON_LOW = 1.0
W_RECON_HIGH = 1.0
W_REFLECT = 0.08
W_SMOOTH_LOW = 0.12
W_SMOOTH_HIGH = 0.12

PATIENCE = 120

SEED = 42
BASE_CHANNELS = 32
LAMBDA_INIT = 1.0