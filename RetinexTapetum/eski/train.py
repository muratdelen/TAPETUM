import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import LOLPairDataset
from model import RetinexTapetumModel
from losses import total_loss_fn
from utils import seed_everything, calc_psnr

def train():

    seed_everything(SEED)

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Dataset
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
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Model
    model = RetinexTapetumModel(BASE_CHANNELS, LAMBDA_INIT).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Scheduler (çok önemli stabilite için)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_psnr = -1

    print("\n===== TRAIN START =====")

    for epoch in range(EPOCHS):

        model.train()

        loss_total = 0

        pbar = tqdm(train_loader)

        for batch in pbar:

            low = batch["low"].to(DEVICE)
            high = batch["high"].to(DEVICE)

            optimizer.zero_grad()

            output = model(low)

            loss, logs = total_loss_fn(output, high)

            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            loss_total += loss.item()

            pbar.set_description(
                f"Epoch {epoch+1}/{EPOCHS} | loss {loss.item():.4f}"
            )

        scheduler.step()

        # validation
        model.eval()

        psnr_total = 0

        with torch.no_grad():

            for batch in val_loader:

                low = batch["low"].to(DEVICE)
                high = batch["high"].to(DEVICE)

                output = model(low)

                psnr = calc_psnr(output["enhanced"], high)

                psnr_total += psnr

        psnr_avg = psnr_total / len(val_loader)

        print(f"\nEpoch {epoch+1} | VAL PSNR: {psnr_avg:.4f}")

        # checkpoint
        if psnr_avg > best_psnr:

            best_psnr = psnr_avg

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_psnr": best_psnr
                },
                os.path.join(CKPT_DIR, "best.pth")
            )

            print("BEST MODEL SAVED")

    print("\nTraining Finished")
    print("Best PSNR:", best_psnr)


if __name__ == "__main__":
    train()