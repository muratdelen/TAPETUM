import os
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from config import DEVICE
from model import RetinexTapetumModel

DATA_ROOT = "../datasets/LoLv2/LOL-v2/Real_captured"
TEST_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")

RUN_ROOT = "../LoLv2/retinex-tapetum"
CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
RESULT_DIR = os.path.join(RUN_ROOT, "results", "Test")

os.makedirs(RESULT_DIR, exist_ok=True)

def list_images(folder):
    return sorted(os.listdir(folder))

def load_model():

    ckpt_path = os.path.join(CKPT_DIR, "best.pth")

    model = RetinexTapetumModel().to(DEVICE)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    return model

@torch.no_grad()
def run():

    model = load_model()

    to_tensor = transforms.ToTensor()

    files = list_images(TEST_LOW_DIR)

    for f in tqdm(files):

        img = Image.open(os.path.join(TEST_LOW_DIR, f)).convert("RGB")

        inp = to_tensor(img).unsqueeze(0).to(DEVICE)

        output = model(inp)

        enh = output["enhanced"]

        out = transforms.ToPILImage()(enh.squeeze().cpu())

        out.save(os.path.join(RESULT_DIR, f))

    print("TEST FINISHED")

if __name__ == "__main__":
    run()