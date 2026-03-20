"""
Training script for RetinexTapetum.

This script follows the same training/reporting structure as the reference
Retinex+Tapetum code so model outputs can be compared more fairly.
"""

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import (
    TRAIN_LOW_DIR,
    TRAIN_HIGH_DIR,
    VAL_LOW_DIR,
    VAL_HIGH_DIR,
    CKPT_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    CROP_SIZE,
    EPOCHS,
    LR,
    SEED,
    BASE_CHANNELS,
    LAMBDA_INIT,
    LAMBDA_MAX,
    PATIENCE,
    GRAD_CLIP_NORM,
)
from dataset import LOLPairDataset
from model import RetinexTapetumModel
from losses import total_loss_fn
from utils import seed_everything, calc_psnr


def train_one_epoch(model, loader, optimizer, device):
    """Run one full training epoch and return averaged logs."""
    model.train()
    running = {
        "total": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "color": 0.0,
        "attn": 0.0,
        "decomp": 0.0,
        "recon_low": 0.0,
        "recon_high": 0.0,
        "reflect": 0.0,
        "smooth_low": 0.0,
        "smooth_high": 0.0,
        "smooth_enh": 0.0,
    }

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False)
    for batch in pbar:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(low, high)
        loss, logs = total_loss_fn(output, low, high)
        loss.backward()

        # Optional stabilization for harder runs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        for k in running:
            running[k] += logs[k]

        pbar.set_postfix(
            {
                "loss": f"{logs['total']:.4f}",
                "decomp": f"{logs['decomp']:.4f}",
                "sEnh": f"{logs['smooth_enh']:.4f}",
                "lam": f"{output['lambda'].item():.3f}",
            }
        )

    n = len(loader)
    for k in running:
        running[k] /= n
    return running


@torch.no_grad()
def validate(model, loader, device):
    """Run validation on the full validation split and return averaged logs."""
    model.eval()
    running = {
        "total": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "color": 0.0,
        "attn": 0.0,
        "decomp": 0.0,
        "recon_low": 0.0,
        "recon_high": 0.0,
        "reflect": 0.0,
        "smooth_low": 0.0,
        "smooth_high": 0.0,
        "smooth_enh": 0.0,
        "psnr": 0.0,
    }

    pbar = tqdm(loader, total=len(loader), desc="Val", leave=False)
    for batch in pbar:
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)

        output = model(low, high)
        _, logs = total_loss_fn(output, low, high)
        psnr = calc_psnr(output["enhanced"], high)

        for k in running:
            if k != "psnr":
                running[k] += logs[k]
        running["psnr"] += psnr

        pbar.set_postfix(
            {
                "loss": f"{logs['total']:.4f}",
                "psnr": f"{psnr:.2f}",
                "decomp": f"{logs['decomp']:.4f}",
                "sEnh": f"{logs['smooth_enh']:.4f}",
                "lam": f"{output['lambda'].item():.3f}",
            }
        )

    n = len(loader)
    for k in running:
        running[k] /= n
    return running


def main():
    seed_everything(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("DEVICE:", DEVICE)
    print("TRAIN_LOW_DIR :", TRAIN_LOW_DIR)
    print("TRAIN_HIGH_DIR:", TRAIN_HIGH_DIR)
    print("VAL_LOW_DIR   :", VAL_LOW_DIR)
    print("VAL_HIGH_DIR  :", VAL_HIGH_DIR)
    print("CKPT_DIR      :", CKPT_DIR)

    train_dataset = LOLPairDataset(
        low_dir=TRAIN_LOW_DIR,
        high_dir=TRAIN_HIGH_DIR,
        crop_size=CROP_SIZE,
        training=True,
    )

    val_dataset = LOLPairDataset(
        low_dir=VAL_LOW_DIR,
        high_dir=VAL_HIGH_DIR,
        crop_size=CROP_SIZE,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
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

        with torch.no_grad():
            lam = (LAMBDA_MAX * torch.sigmoid(model.lambda_param)).item()

        # First append current epoch history, then write checkpoints.
        # This avoids checkpoints lagging one epoch behind in their history field.
        history.append({
            "epoch": epoch,
            "train": train_logs,
            "val": val_logs,
            "lambda": lam,
        })

        log_str = (
            f"Epoch {epoch:03d} | "
            f"train_loss {train_logs['total']:.4f} | "
            f"val_loss {val_logs['total']:.4f} | "
            f"val_psnr {val_logs['psnr']:.2f} | "
            f"decomp {val_logs['decomp']:.4f} | "
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
    main()
