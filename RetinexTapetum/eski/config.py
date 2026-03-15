import os
import torch

DATA_ROOT = "../datasets/LoLv2/LOL-v2/Real_captured"

TRAIN_LOW_DIR  = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")
VAL_LOW_DIR    = os.path.join(DATA_ROOT, "Test", "Low")
VAL_HIGH_DIR   = os.path.join(DATA_ROOT, "Test", "Normal")

RUN_ROOT = "../LoLv2/retinex-tapetum"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BATCH_SIZE = 8
NUM_WORKERS = 2
CROP_SIZE = 256
EPOCHS = 120
LR = 2e-4
SEED = 42

BASE_CHANNELS = 32
LAMBDA_INIT = 1.0