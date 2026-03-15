"""
Dataset definition for LOL-v2 paired low/high images.

This file enforces the same data protocol used by the reference training code:
- low and high images are paired by filename,
- training mode applies resize-if-needed, random crop and augmentation,
- validation mode keeps the full image without random augmentation.
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_EXTS


def list_images(folder):
    """Return sorted image filenames inside a folder."""
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])


class LOLPairDataset(Dataset):
    """
    Paired LOL dataset.

    Each sample returns:
        {
            "low":  low-light RGB tensor in [0,1],
            "high": normal-light RGB tensor in [0,1],
            "name": original filename
        }

    Parameters
    ----------
    low_dir : str
        Folder containing low-light images.
    high_dir : str
        Folder containing corresponding normal-light images.
    crop_size : int
        Random crop size used only during training.
    training : bool
        If True, random crop and augmentation are applied.
    """

    def __init__(self, low_dir, high_dir, crop_size=256, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.crop_size = crop_size
        self.training = training
        self.to_tensor = transforms.ToTensor()

        low_files = list_images(low_dir)
        high_files = set(list_images(high_dir))

        # Only keep fully paired samples. This avoids training/evaluation errors.
        self.files = [f for f in low_files if f in high_files]

        print(f"[Dataset] {low_dir}")
        print(f"Paired samples: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def _resize_if_needed(self, low, high):
        """Upscale small images so random crop never fails."""
        w, h = low.size
        if w < self.crop_size or h < self.crop_size:
            new_w = max(w, self.crop_size)
            new_h = max(h, self.crop_size)
            low = low.resize((new_w, new_h), Image.BICUBIC)
            high = high.resize((new_w, new_h), Image.BICUBIC)
        return low, high

    def _random_crop(self, low, high):
        """Apply the same crop coordinates to low and high images."""
        w, h = low.size
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        low = low.crop((x, y, x + self.crop_size, y + self.crop_size))
        high = high.crop((x, y, x + self.crop_size, y + self.crop_size))
        return low, high

    def _augment(self, low, high):
        """Apply paired flip/rotation augmentation."""
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
