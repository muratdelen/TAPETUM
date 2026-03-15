"""Training script for the standardized RetinexTapetum model."""

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import LOLPairDataset
from model import RetinexTapetumModel
from losses import total_loss_fn
from utils import seed_everything, calc_psnr, write_history_csv


def train_one_epoch(model, loader, optimizer, device):
    """Run one training epoch and return averaged loss logs."""
    model.train()
    running = {"total": 0.0, "l1": 0.0, "ssim": 0.0, "color": 0.0, "attn": 0.0, "smooth_enh": 0.0}

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False)
    for batch in pbar:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(low)
        loss, logs = total_loss_fn(output, low, high)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        for k in running:
            running[k] += logs[k]

        pbar.set_postfix({
            "loss": f"{logs['total']:.4f}",
            "l1": f"{logs['l1']:.4f}",
            "ssim": f"{logs['ssim']:.4f}",
            "sEnh": f"{logs['smooth_enh']:.4f}",
            "lam": f"{output['lambda'].item():.3f}",
        })

    n = len(loader)
    for k in running:
        running[k] /= n
    return running


@torch.no_grad()
def validate(model, loader, device):
    """Run validation and return averaged logs plus PSNR."""
    model.eval()
    running = {
        "total": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "color": 0.0,
        "attn": 0.0,
        "smooth_enh": 0.0,
        "psnr": 0.0,
    }

    pbar = tqdm(loader, total=len(loader), desc="Val", leave=False)
    for batch in pbar:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        output = model(low)
        loss, logs = total_loss_fn(output, low, high)
        psnr = calc_psnr(output["enhanced"], high)

        for k in logs:
            running[k] += logs[k]
        running["psnr"] += psnr

        pbar.set_postfix({
            "loss": f"{logs['total']:.4f}",
            "psnr": f"{psnr:.2f}",
            "sEnh": f"{logs['smooth_enh']:.4f}",
            "lam": f"{output['lambda'].item():.3f}",
        })

    n = len(loader)
    for k in running:
        running[k] /= n
    return running


def train():
    """Main training entry point."""
    seed_everything(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RUN_ROOT, exist_ok=True)

    print("DEVICE:", DEVICE)
    print("TRAIN_LOW_DIR :", TRAIN_LOW_DIR)
    print("TRAIN_HIGH_DIR:", TRAIN_HIGH_DIR)
    print("VAL_LOW_DIR   :", VAL_LOW_DIR)
    print("VAL_HIGH_DIR  :", VAL_HIGH_DIR)
    print("CKPT_DIR      :", CKPT_DIR)

    train_dataset = LOLPairDataset(TRAIN_LOW_DIR, TRAIN_HIGH_DIR, CROP_SIZE, True)
    val_dataset = LOLPairDataset(VAL_LOW_DIR, VAL_HIGH_DIR, CROP_SIZE, False)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("Train batches:", len(train_loader))
    print("Val batches  :", len(val_loader))

    model = RetinexTapetumModel(
        base=BASE_CHANNELS,
        lambda_init=LAMBDA_INIT,
        lambda_max=LAMBDA_MAX,
        gaussian_kernel_size=GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma=GAUSSIAN_SIGMA,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", f"{num_params:,}")

    best_psnr = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    print("\n===== TRAIN START =====")
    for epoch in range(1, EPOCHS + 1):
        train_logs = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_logs = validate(model, val_loader, DEVICE)
        scheduler.step()
        lam = output_lambda = (LAMBDA_MAX * torch.sigmoid(model.lambda_param)).item()

        history_row = {
            "epoch": epoch,
            "train_total": train_logs["total"],
            "train_l1": train_logs["l1"],
            "train_ssim": train_logs["ssim"],
            "train_color": train_logs["color"],
            "train_attn": train_logs["attn"],
            "train_smooth_enh": train_logs["smooth_enh"],
            "val_total": val_logs["total"],
            "val_l1": val_logs["l1"],
            "val_ssim": val_logs["ssim"],
            "val_color": val_logs["color"],
            "val_attn": val_logs["attn"],
            "val_smooth_enh": val_logs["smooth_enh"],
            "val_psnr": val_logs["psnr"],
            "lambda": lam,
        }
        history.append(history_row)
        write_history_csv(history, HISTORY_CSV)

        log_str = (
            f"Epoch {epoch:03d} | "
            f"train_loss {train_logs['total']:.4f} | "
            f"val_loss {val_logs['total']:.4f} | "
            f"val_psnr {val_logs['psnr']:.2f} | "
            f"smooth_enh {val_logs['smooth_enh']:.4f} | "
            f"lambda {lam:.4f}"
        )
        print(log_str)

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "best_epoch": best_epoch,
            "history": history,
            "lambda": lam,
        }
        torch.save(ckpt, os.path.join(CKPT_DIR, "last.pth"))

        if val_logs["psnr"] > best_psnr:
            best_psnr = val_logs["psnr"]
            best_epoch = epoch
            no_improve = 0
            ckpt["best_psnr"] = best_psnr
            ckpt["best_epoch"] = best_epoch
            torch.save(ckpt, os.path.join(CKPT_DIR, "best.pth"))
            print(f"✅ Best model updated -> Epoch: {best_epoch}, PSNR: {best_psnr:.4f}")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"🛑 Early stopping at epoch {epoch}")
            print(f"Best epoch: {best_epoch} | Best PSNR: {best_psnr:.4f}")
            break

    print("===== TRAIN FINISHED =====")
    print("Best epoch:", best_epoch)
    print("Best PSNR:", best_psnr)
    print("Best checkpoint:", os.path.join(CKPT_DIR, "best.pth"))
    print("Last checkpoint:", os.path.join(CKPT_DIR, "last.pth"))


if __name__ == "__main__":
    train()
