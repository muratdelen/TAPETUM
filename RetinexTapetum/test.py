"""Inference script for the standardized RetinexTapetum model."""

import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from config import DEVICE, TEST_LOW_DIR, CKPT_DIR, RESULT_DIR, BASE_CHANNELS, LAMBDA_INIT, LAMBDA_MAX, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA
from model import RetinexTapetumModel


os.makedirs(RESULT_DIR, exist_ok=True)


def list_images(folder):
    """Return sorted test file names."""
    return sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


def load_model():
    """Load the best checkpoint saved during training."""
    ckpt_path = os.path.join(CKPT_DIR, "best.pth")
    model = RetinexTapetumModel(
        base=BASE_CHANNELS,
        lambda_init=LAMBDA_INIT,
        lambda_max=LAMBDA_MAX,
        gaussian_kernel_size=GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma=GAUSSIAN_SIGMA,
    ).to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def run():
    """Run inference on the LOL-v2 test low-light folder and save outputs."""
    print("DEVICE:", DEVICE)
    print("TEST_LOW_DIR:", TEST_LOW_DIR)
    print("RESULT_DIR  :", RESULT_DIR)

    model = load_model()
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    files = list_images(TEST_LOW_DIR)
    for fname in tqdm(files, desc="Test"):
        img = Image.open(os.path.join(TEST_LOW_DIR, fname)).convert("RGB")
        inp = to_tensor(img).unsqueeze(0).to(DEVICE)
        output = model(inp)
        enh = output["enhanced"].squeeze(0).cpu().clamp(0.0, 1.0)
        out = to_pil(enh)
        out.save(os.path.join(RESULT_DIR, fname))

    print("TEST FINISHED")


if __name__ == "__main__":
    run()
