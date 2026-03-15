# =========================
# RETINEX + TAPETUM TRAIN
# Single-cell Google Colab code
# =========================

# ===== 1) Imports =====
import os
import math
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ===== 2) Seed =====
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

# ===== 3) Paths =====
DATA_ROOT = "../datasets/LoLv2/LOL-v2/Real_captured"
TRAIN_LOW_DIR  = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")
VAL_LOW_DIR    = os.path.join(DATA_ROOT, "Test", "Low")
VAL_HIGH_DIR   = os.path.join(DATA_ROOT, "Test", "Normal")

RUN_ROOT = "../LoLv2/retinex-tapetum"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

print("TRAIN_LOW_DIR :", TRAIN_LOW_DIR)
print("TRAIN_HIGH_DIR:", TRAIN_HIGH_DIR)
print("VAL_LOW_DIR   :", VAL_LOW_DIR)
print("VAL_HIGH_DIR  :", VAL_HIGH_DIR)
print("CKPT_DIR      :", CKPT_DIR)

# ===== 4) Hyperparameters =====
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BATCH_SIZE = 8
NUM_WORKERS = 2
CROP_SIZE = 256
EPOCHS = 120
LR = 2e-4

# ===== 5) Dataset =====
def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])

class LOLPairDataset(Dataset):
    def __init__(self, low_dir, high_dir, crop_size=256, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.crop_size = crop_size
        self.training = training
        self.to_tensor = transforms.ToTensor()

        low_files = list_images(low_dir)
        high_files = set(list_images(high_dir))
        self.files = [f for f in low_files if f in high_files]

        print(f"[Dataset] {low_dir}")
        print(f"Paired samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def _resize_if_needed(self, low, high):
        w, h = low.size
        if w < self.crop_size or h < self.crop_size:
            new_w = max(w, self.crop_size)
            new_h = max(h, self.crop_size)
            low = low.resize((new_w, new_h), Image.BICUBIC)
            high = high.resize((new_w, new_h), Image.BICUBIC)
        return low, high

    def _random_crop(self, low, high):
        w, h = low.size
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        low = low.crop((x, y, x + self.crop_size, y + self.crop_size))
        high = high.crop((x, y, x + self.crop_size, y + self.crop_size))
        return low, high

    def _augment(self, low, high):
        if random.random() < 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        if k:
            low = low.rotate(90 * k)
            high = high.rotate(90 * k)
        return low, high

    def __getitem__(self, idx):
        fname = self.files[idx]
        low = Image.open(os.path.join(self.low_dir, fname)).convert("RGB")
        high = Image.open(os.path.join(self.high_dir, fname)).convert("RGB")

        if self.training:
            low, high = self._resize_if_needed(low, high)
            low, high = self._random_crop(low, high)
            low, high = self._augment(low, high)

        low = self.to_tensor(low)
        high = self.to_tensor(high)

        return {"low": low, "high": high, "name": fname}

# ===== 6) Retinex decomposition =====
def gaussian_kernel(kernel_size=15, sigma=3.0, channels=3, device="cpu"):
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel

def gaussian_blur(x, kernel_size=15, sigma=3.0):
    c = x.shape[1]
    kernel = gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=c, device=x.device)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(x, kernel, groups=c)
    return out

def retinex_decompose(x):
    # x in [0,1]
    L = gaussian_blur(x, kernel_size=21, sigma=5.0)
    R = x / (L + 1e-6)
    R = torch.clamp(R, 0.0, 5.0)
    return R, L

# ===== 7) Model =====
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class TapetumAttentionNet(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_conv = nn.Conv2d(base, 3, 1)

    def forward(self, L):
        e1 = self.enc1(L)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        raw = self.out_conv(d1)

        # T = sigmoid(f(L)) * (1 - L)
        T = torch.sigmoid(raw) * (1.0 - L)
        return T

class RetinexTapetumModel(nn.Module):
    def __init__(self, base=32, lambda_init=1.0):
        super().__init__()
        self.tapetum_net = TapetumAttentionNet(in_ch=3, base=base)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))

    def forward(self, x):
        # I = R * L
        R, L = retinex_decompose(x)

        # T = sigmoid(f(L)) * (1 - L)
        T = self.tapetum_net(L)

        # L_t = L * (1 + lambda * T)
        lam = F.softplus(self.lambda_param)
        L_t = L * (1.0 + lam * T)

        # I_enh = R * L_t
        I_enh = R * L_t
        I_enh = torch.clamp(I_enh, 0.0, 1.0)

        return {
            "enhanced": I_enh,
            "reflectance": R,
            "illumination": L,
            "tapetum_attention": T,
            "illumination_t": L_t,
            "lambda": lam
        }

# ===== 8) Loss =====
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

def create_gaussian_window(window_size, channel, device):
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()

def ssim_loss(pred, target, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channel = pred.size(1)
    window = create_gaussian_window(window_size, channel, pred.device)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
    )
    return 1 - ssim_map.mean()

def color_consistency_loss(pred, target):
    pred_mean = pred.mean(dim=[2, 3])
    target_mean = target.mean(dim=[2, 3])
    return F.l1_loss(pred_mean, target_mean)

def attention_regularization(T):
    return T.mean()

def total_loss_fn(output, gt, w_l1=1.0, w_ssim=0.5, w_color=0.1, w_attn=0.01):
    pred = output["enhanced"]
    T = output["tapetum_attention"]

    l1 = charbonnier_loss(pred, gt)
    ssim_l = ssim_loss(pred, gt)
    color_l = color_consistency_loss(pred, gt)
    attn_l = attention_regularization(T)

    total = w_l1 * l1 + w_ssim * ssim_l + w_color * color_l + w_attn * attn_l
    logs = {
        "total": total.item(),
        "l1": l1.item(),
        "ssim": ssim_l.item(),
        "color": color_l.item(),
        "attn": attn_l.item(),
    }
    return total, logs

def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse)

# ===== 9) DataLoaders =====
train_dataset = LOLPairDataset(
    low_dir=TRAIN_LOW_DIR,
    high_dir=TRAIN_HIGH_DIR,
    crop_size=CROP_SIZE,
    training=True
)

val_dataset = LOLPairDataset(
    low_dir=VAL_LOW_DIR,
    high_dir=VAL_HIGH_DIR,
    crop_size=CROP_SIZE,
    training=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("Train batches:", len(train_loader))
print("Val batches  :", len(val_loader))

# ===== 10) Model/Optimizer =====
model = RetinexTapetumModel(base=32, lambda_init=1.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params:", f"{num_params:,}")

# ===== 11) Train/Val functions =====
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "color": 0.0, "attn": 0.0, "psnr": 0.0}

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False)
    for batch in pbar:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(low)
        loss, logs = total_loss_fn(output, high)
        loss.backward()
        optimizer.step()

        psnr = calc_psnr(output["enhanced"].detach(), high)

        running["total"] += logs["total"]
        running["l1"] += logs["l1"]
        running["ssim"] += logs["ssim"]
        running["color"] += logs["color"]
        running["attn"] += logs["attn"]
        running["psnr"] += psnr

        pbar.set_postfix({
            "loss": f"{logs['total']:.4f}",
            "psnr": f"{psnr:.2f}",
            "lam": f"{output['lambda'].item():.3f}"
        })

    n = len(loader)
    for k in running:
        running[k] /= n
    return running

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "color": 0.0, "attn": 0.0, "psnr": 0.0}

    pbar = tqdm(loader, total=len(loader), desc="Val", leave=False)
    for batch in pbar:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        output = model(low)
        loss, logs = total_loss_fn(output, high)
        psnr = calc_psnr(output["enhanced"], high)

        running["total"] += logs["total"]
        running["l1"] += logs["l1"]
        running["ssim"] += logs["ssim"]
        running["color"] += logs["color"]
        running["attn"] += logs["attn"]
        running["psnr"] += psnr

        pbar.set_postfix({
            "loss": f"{logs['total']:.4f}",
            "psnr": f"{psnr:.2f}",
            "lam": f"{output['lambda'].item():.3f}"
        })

    n = len(loader)
    for k in running:
        running[k] /= n
    return running

# ===== 12) Training loop =====
best_psnr = -1.0
history = []

print("\n===== TRAIN START =====")
for epoch in range(1, EPOCHS + 1):
    train_logs = train_one_epoch(model, train_loader, optimizer, device)
    val_logs = validate(model, val_loader, device)
    scheduler.step()

    log_str = (
        f"Epoch {epoch:03d} | "
        f"train_loss {train_logs['total']:.4f} | "
        f"train_psnr {train_logs['psnr']:.2f} | "
        f"val_loss {val_logs['total']:.4f} | "
        f"val_psnr {val_logs['psnr']:.2f} | "
        f"lambda {F.softplus(model.lambda_param).item():.4f}"
    )
    print(log_str)

    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_psnr": best_psnr,
        "history": history,
    }
    torch.save(ckpt, os.path.join(CKPT_DIR, "last.pth"))

    if val_logs["psnr"] > best_psnr:
        best_psnr = val_logs["psnr"]
        ckpt["best_psnr"] = best_psnr
        torch.save(ckpt, os.path.join(CKPT_DIR, "best.pth"))
        print(f"✅ Best model updated -> PSNR: {best_psnr:.4f}")

    history.append({
        "epoch": epoch,
        "train": train_logs,
        "val": val_logs,
        "lambda": F.softplus(model.lambda_param).item()
    })

print("===== TRAIN FINISHED =====")
print("Best PSNR:", best_psnr)
print("Best checkpoint:", os.path.join(CKPT_DIR, "best.pth"))
print("Last checkpoint:", os.path.join(CKPT_DIR, "last.pth"))
