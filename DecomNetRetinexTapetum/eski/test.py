import os
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from config import (
    DEVICE, TEST_LOW_DIR, CKPT_DIR, RESULT_DIR,
    BASE_CHANNELS, LAMBDA_INIT, LAMBDA_MAX
)
from model import DecomNetRetinexTapetumModel

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

os.makedirs(RESULT_DIR, exist_ok=True)

print("DEVICE      :", DEVICE)
print("TEST_LOW_DIR:", TEST_LOW_DIR)
print("CKPT_DIR    :", CKPT_DIR)
print("RESULT_DIR  :", RESULT_DIR)

def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])

def load_model():
    ckpt_path = os.path.join(CKPT_DIR, "best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = DecomNetRetinexTapetumModel(
        base=BASE_CHANNELS,
        lambda_init=LAMBDA_INIT,
        lambda_max=LAMBDA_MAX
    ).to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    if "best_psnr" in checkpoint:
        print(f"Best PSNR: {checkpoint['best_psnr']:.4f}")
    if "best_epoch" in checkpoint:
        print(f"Best Epoch: {checkpoint['best_epoch']}")
    if "lambda" in checkpoint:
        print("Saved lambda:", checkpoint["lambda"])

    return model

def tensor_to_pil(x):
    x = x.squeeze(0).detach().cpu().clamp(0.0, 1.0)
    return transforms.ToPILImage()(x)

@torch.no_grad()
def run_test():
    model = load_model()
    to_tensor = transforms.ToTensor()

    files = list_images(TEST_LOW_DIR)
    print("Test image count:", len(files))

    for fname in tqdm(files, desc="Testing"):
        in_path = os.path.join(TEST_LOW_DIR, fname)
        out_path = os.path.join(RESULT_DIR, fname)

        img = Image.open(in_path).convert("RGB")
        inp = to_tensor(img).unsqueeze(0).to(DEVICE)

        output = model(inp)
        enh = output["enhanced"]

        out_img = tensor_to_pil(enh)
        out_img.save(out_path)

    print("===== TEST FINISHED =====")
    print("Results saved to:", RESULT_DIR)

if __name__ == "__main__":
    run_test()