# ============================================================
# TAPETUM / LOLv2 - FULL METRICS + VISUAL REPORT GENERATOR
# Uses RetinexNet precomputed CSV metrics if available
# ============================================================

import os
import math
import subprocess
import sys

def ensure_package(pkg_name, import_name=None):
    import_name = import_name or pkg_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])

ensure_package("lpips")
ensure_package("openpyxl")
ensure_package("xlsxwriter")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import cv2
import torch
import lpips
from skimage.metrics import structural_similarity as ssim

ROOT = "/content/TAPETUM"

DATA_ROOT = os.path.join(ROOT, "datasets", "LoLv2", "LOL-v2", "Real_captured")
TEST_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")
GT_DIR       = os.path.join(DATA_ROOT, "Test", "Normal")

MODEL_DIRS = {
    "RetinexNet": os.path.join(ROOT, "RetinexNet", "test_results_lolv2"),
    "RetinexTapetum": os.path.join(ROOT, "LoLv2", "retinex-tapetum", "results", "Test"),
    "RetinexTapetumRGB": os.path.join(ROOT, "LoLv2", "RetinexTapetumRGB", "results", "Test"),
    "DecomNetRetinexTapetum": os.path.join(ROOT, "LoLv2", "DecomNetRetinexTapetum", "results", "Test"),
    "DecomNetRetinexTapetumRGB": os.path.join(ROOT, "LoLv2", "DecomNetRetinexTapetumRGB", "results", "Test"),
}

# Optional precomputed CSVs
RETINEXNET_METRICS_CSV = os.path.join(ROOT, "RetinexNet", "retinexnet_test_metrics.csv")
RETINEXNET_SUMMARY_CSV = os.path.join(ROOT, "RetinexNet", "retinexnet_test_metrics_summary.csv")

METRICS_ROOT = os.path.join(ROOT, "Metrics")
TABLE_DIR    = os.path.join(METRICS_ROOT, "tables")
VIS_DIR      = os.path.join(METRICS_ROOT, "visuals")
BEST_DIR     = os.path.join(VIS_DIR, "best_cases")
WORST_DIR    = os.path.join(VIS_DIR, "worst_cases")
ALL_VIS_DIR  = os.path.join(VIS_DIR, "all_panels")

for d in [METRICS_ROOT, TABLE_DIR, VIS_DIR, BEST_DIR, WORST_DIR, ALL_VIS_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def list_images(folder):
    if not os.path.exists(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])

def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_to_gt(pred, gt):
    if pred.shape[:2] != gt.shape[:2]:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
    return pred

def calc_mse(img1, img2):
    return float(np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2))

def calc_mae(img1, img2):
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))

def calc_rmse(img1, img2):
    return float(np.sqrt(calc_mse(img1, img2)))

def calc_psnr(img1, img2):
    mse = calc_mse(img1, img2)
    if mse == 0:
        return 100.0
    return float(20 * math.log10(255.0 / math.sqrt(mse)))

def calc_ssim(img1, img2):
    return float(ssim(img1, img2, channel_axis=2, data_range=255))

def np_to_lpips_tensor(img, device):
    x = torch.from_numpy(img).float() / 127.5 - 1.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    return x

def safe_filename(name):
    return name.replace("/", "_").replace("\\", "_")

def get_font(size=18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def add_text_block(draw, x, y, text, font, fill=(255,255,255), bg=(0,0,0)):
    lines = text.split("\n")
    widths, heights = [], []
    for line in lines:
        bbox = draw.textbbox((0,0), line, font=font)
        widths.append(bbox[2]-bbox[0])
        heights.append(bbox[3]-bbox[1])
    w = max(widths) + 12
    h = sum(heights) + 6 * len(lines) + 8
    draw.rectangle([x, y, x+w, y+h], fill=bg)
    yy = y + 4
    for line in lines:
        draw.text((x+6, yy), line, font=font, fill=fill)
        bbox = draw.textbbox((0,0), line, font=font)
        yy += (bbox[3]-bbox[1]) + 6
    return w, h

def pil_from_np(img):
    return Image.fromarray(img.astype(np.uint8))

def make_panel(input_img, gt_img, preds_dict, metrics_dict, save_path, title=None):
    font_title = get_font(20)
    font_body = get_font(16)

    block_w = 320
    block_h = 260
    margin = 20
    top_text_h = 40

    items = [("Input", input_img, None), ("GT", gt_img, None)]
    for model_name, img in preds_dict.items():
        items.append((model_name, img, metrics_dict.get(model_name)))

    cols = 2
    rows = math.ceil(len(items) / cols)

    canvas_w = cols * block_w + (cols + 1) * margin
    canvas_h = rows * block_h + (rows + 1) * margin + top_text_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    if title is None:
        title = os.path.basename(save_path)
    draw.text((margin, 8), title, font=font_title, fill=(255,255,255))

    for idx, (name, img_np, metric) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * (block_w + margin)
        y0 = margin + top_text_h + r * (block_h + margin)

        draw.rectangle([x0, y0, x0+block_w, y0+block_h], outline=(180,180,180), width=2)

        img_pil = pil_from_np(img_np)
        img_pil.thumbnail((block_w-20, block_h-90))
        ix = x0 + (block_w - img_pil.width) // 2
        iy = y0 + 38
        canvas.paste(img_pil, (ix, iy))

        draw.text((x0+10, y0+8), name, font=font_body, fill=(255,255,0))

        if metric is not None:
            metric_text = (
                f"PSNR:  {metric.get('psnr', float('nan')):.3f}\n"
                f"SSIM:  {metric.get('ssim', float('nan')):.4f}\n"
                f"MAE:   {metric.get('mae', float('nan')):.4f}\n"
                f"MSE:   {metric.get('mse', float('nan')):.4f}\n"
                f"RMSE:  {metric.get('rmse', float('nan')):.4f}\n"
                f"LPIPS: {metric.get('lpips', float('nan')):.4f}"
            )
            add_text_block(draw, x0+10, y0+block_h-88, metric_text, font_body)

    canvas.save(save_path)

def normalize_precomputed_retinexnet(df):
    df = df.copy()
    rename_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # image column
    for candidate in ["image", "filename", "file", "name"]:
        if candidate in lower_cols:
            rename_map[lower_cols[candidate]] = "image"
            break

    # metric columns
    for m in ["psnr", "ssim", "mae", "mse", "rmse", "lpips"]:
        if m in lower_cols:
            rename_map[lower_cols[m]] = m

    df = df.rename(columns=rename_map)

    needed = ["image", "psnr", "ssim", "mae", "mse"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"RetinexNet metrics CSV missing columns: {missing}")

    if "rmse" not in df.columns:
        df["rmse"] = np.sqrt(df["mse"].astype(float))

    if "lpips" not in df.columns:
        df["lpips"] = np.nan

    df["model"] = "RetinexNet"
    return df[["image", "model", "psnr", "ssim", "mae", "mse", "rmse", "lpips"]]

print("Using optional RetinexNet CSVs if available:")
print(" -", RETINEXNET_METRICS_CSV)
print(" -", RETINEXNET_SUMMARY_CSV)

device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()

gt_files = list_images(GT_DIR)
low_files = set(list_images(TEST_LOW_DIR))
if len(gt_files) == 0:
    raise FileNotFoundError(f"No GT images found: {GT_DIR}")
gt_files = [f for f in gt_files if f in low_files]

all_rows = []
per_image_scores = {fname: {} for fname in gt_files}

# 1) RetinexNet from precomputed CSV if possible
retinexnet_loaded_from_csv = False
if os.path.exists(RETINEXNET_METRICS_CSV):
    try:
        df_retinex_pre = pd.read_csv(RETINEXNET_METRICS_CSV)
        df_retinex_pre = normalize_precomputed_retinexnet(df_retinex_pre)
        df_retinex_pre = df_retinex_pre[df_retinex_pre["image"].isin(gt_files)].copy()

        for _, row in df_retinex_pre.iterrows():
            row_dict = {
                "image": row["image"],
                "model": "RetinexNet",
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "mae": float(row["mae"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "lpips": float(row["lpips"]) if pd.notna(row["lpips"]) else np.nan,
            }
            all_rows.append(row_dict)
            per_image_scores[row["image"]]["RetinexNet"] = row_dict

        retinexnet_loaded_from_csv = len(df_retinex_pre) > 0
        print(f"Loaded RetinexNet metrics from CSV: {len(df_retinex_pre)} rows")
    except Exception as e:
        print("Failed to use RetinexNet precomputed CSV, falling back to image evaluation.")
        print("Reason:", e)

# 2) Evaluate models from images
for model_name, pred_dir in MODEL_DIRS.items():
    if model_name == "RetinexNet" and retinexnet_loaded_from_csv:
        continue

    if not os.path.exists(pred_dir):
        print(f"Skipped missing folder: {pred_dir}")
        continue

    pred_files = set(list_images(pred_dir))
    matched_files = [f for f in gt_files if f in pred_files]

    for fname in tqdm(matched_files, desc=f"Eval-{model_name}"):
        gt = read_rgb(os.path.join(GT_DIR, fname))
        pred = read_rgb(os.path.join(pred_dir, fname))
        if gt is None or pred is None:
            continue

        pred = resize_to_gt(pred, gt)

        psnr_val = calc_psnr(pred, gt)
        ssim_val = calc_ssim(pred, gt)
        mae_val  = calc_mae(pred, gt)
        mse_val  = calc_mse(pred, gt)
        rmse_val = calc_rmse(pred, gt)

        with torch.no_grad():
            gt_t = np_to_lpips_tensor(gt, device)
            pred_t = np_to_lpips_tensor(pred, device)
            lpips_val = float(lpips_model(pred_t, gt_t).item())

        row = {
            "image": fname,
            "model": model_name,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "mae": mae_val,
            "mse": mse_val,
            "rmse": rmse_val,
            "lpips": lpips_val,
        }
        all_rows.append(row)
        per_image_scores[fname][model_name] = row

df_all = pd.DataFrame(all_rows)
if df_all.empty:
    raise RuntimeError("No metrics computed. Check model result folders.")

df_summary = (
    df_all.groupby("model", as_index=False)
    .agg(
        matched_files=("image", "count"),
        psnr=("psnr", "mean"),
        ssim=("ssim", "mean"),
        mae=("mae", "mean"),
        mse=("mse", "mean"),
        rmse=("rmse", "mean"),
        lpips=("lpips", "mean"),
    )
    .sort_values(by=["psnr", "ssim"], ascending=[False, False])
    .reset_index(drop=True)
)

# If RetinexNet summary CSV exists, also save a copy for reference
if os.path.exists(RETINEXNET_SUMMARY_CSV):
    try:
        ref_summary = pd.read_csv(RETINEXNET_SUMMARY_CSV)
        ref_summary.to_csv(os.path.join(TABLE_DIR, "retinexnet_original_summary_reference.csv"), index=False, encoding="utf-8-sig")
    except Exception as e:
        print("Could not copy RetinexNet summary CSV reference:", e)

winner_counter = {
    model: {
        "best_psnr_count": 0,
        "best_ssim_count": 0,
        "best_mae_count": 0,
        "best_mse_count": 0,
        "best_rmse_count": 0,
        "best_lpips_count": 0,
    }
    for model in MODEL_DIRS.keys()
}

for fname, scores in per_image_scores.items():
    if not scores:
        continue

    def valid_metric_items(metric_name, prefer_lower=False):
        items = []
        for m, vals in scores.items():
            val = vals.get(metric_name, np.nan)
            if pd.notna(val):
                items.append((m, val))
        return items

    items = valid_metric_items("psnr")
    if items: winner_counter[max(items, key=lambda x: x[1])[0]]["best_psnr_count"] += 1
    items = valid_metric_items("ssim")
    if items: winner_counter[max(items, key=lambda x: x[1])[0]]["best_ssim_count"] += 1
    items = valid_metric_items("mae")
    if items: winner_counter[min(items, key=lambda x: x[1])[0]]["best_mae_count"] += 1
    items = valid_metric_items("mse")
    if items: winner_counter[min(items, key=lambda x: x[1])[0]]["best_mse_count"] += 1
    items = valid_metric_items("rmse")
    if items: winner_counter[min(items, key=lambda x: x[1])[0]]["best_rmse_count"] += 1
    items = valid_metric_items("lpips")
    if items: winner_counter[min(items, key=lambda x: x[1])[0]]["best_lpips_count"] += 1

df_winners = pd.DataFrame([
    {"model": model, **vals}
    for model, vals in winner_counter.items()
]).sort_values(
    by=["best_psnr_count", "best_ssim_count", "best_lpips_count"],
    ascending=[False, False, False]
).reset_index(drop=True)

pivot_psnr  = df_all.pivot(index="image", columns="model", values="psnr").reset_index()
pivot_ssim  = df_all.pivot(index="image", columns="model", values="ssim").reset_index()
pivot_mae   = df_all.pivot(index="image", columns="model", values="mae").reset_index()
pivot_mse   = df_all.pivot(index="image", columns="model", values="mse").reset_index()
pivot_rmse  = df_all.pivot(index="image", columns="model", values="rmse").reset_index()
pivot_lpips = df_all.pivot(index="image", columns="model", values="lpips").reset_index()

rank_df = df_summary.copy()
rank_df["rank_psnr"] = rank_df["psnr"].rank(ascending=False, method="min")
rank_df["rank_ssim"] = rank_df["ssim"].rank(ascending=False, method="min")
rank_df["rank_mae"] = rank_df["mae"].rank(ascending=True, method="min")
rank_df["rank_mse"] = rank_df["mse"].rank(ascending=True, method="min")
rank_df["rank_rmse"] = rank_df["rmse"].rank(ascending=True, method="min")
# NaN LPIPS should rank worst
rank_df["lpips_fill"] = rank_df["lpips"].fillna(rank_df["lpips"].max() if rank_df["lpips"].notna().any() else 999999)
rank_df["rank_lpips"] = rank_df["lpips_fill"].rank(ascending=True, method="min")
rank_df["rank_total"] = rank_df[
    ["rank_psnr","rank_ssim","rank_mae","rank_mse","rank_rmse","rank_lpips"]
].sum(axis=1)
rank_df = rank_df.sort_values(by="rank_total", ascending=True).reset_index(drop=True)
rank_df = rank_df.drop(columns=["lpips_fill"])

df_all.to_csv(os.path.join(TABLE_DIR, "all_image_metrics.csv"), index=False, encoding="utf-8-sig")
df_summary.to_csv(os.path.join(TABLE_DIR, "summary_metrics.csv"), index=False, encoding="utf-8-sig")
df_winners.to_csv(os.path.join(TABLE_DIR, "winner_counts.csv"), index=False, encoding="utf-8-sig")
rank_df.to_csv(os.path.join(TABLE_DIR, "model_ranking.csv"), index=False, encoding="utf-8-sig")
pivot_psnr.to_csv(os.path.join(TABLE_DIR, "pivot_psnr.csv"), index=False, encoding="utf-8-sig")
pivot_ssim.to_csv(os.path.join(TABLE_DIR, "pivot_ssim.csv"), index=False, encoding="utf-8-sig")
pivot_mae.to_csv(os.path.join(TABLE_DIR, "pivot_mae.csv"), index=False, encoding="utf-8-sig")
pivot_mse.to_csv(os.path.join(TABLE_DIR, "pivot_mse.csv"), index=False, encoding="utf-8-sig")
pivot_rmse.to_csv(os.path.join(TABLE_DIR, "pivot_rmse.csv"), index=False, encoding="utf-8-sig")
pivot_lpips.to_csv(os.path.join(TABLE_DIR, "pivot_lpips.csv"), index=False, encoding="utf-8-sig")

excel_path = os.path.join(TABLE_DIR, "paper_metrics_report.xlsx")
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df_summary.to_excel(writer, sheet_name="summary", index=False)
    df_winners.to_excel(writer, sheet_name="winner_counts", index=False)
    rank_df.to_excel(writer, sheet_name="ranking", index=False)
    df_all.to_excel(writer, sheet_name="all_metrics", index=False)
    pivot_psnr.to_excel(writer, sheet_name="pivot_psnr", index=False)
    pivot_ssim.to_excel(writer, sheet_name="pivot_ssim", index=False)
    pivot_mae.to_excel(writer, sheet_name="pivot_mae", index=False)
    pivot_mse.to_excel(writer, sheet_name="pivot_mse", index=False)
    pivot_rmse.to_excel(writer, sheet_name="pivot_rmse", index=False)
    pivot_lpips.to_excel(writer, sheet_name="pivot_lpips", index=False)

with open(os.path.join(TABLE_DIR, "summary_table_latex.txt"), "w", encoding="utf-8") as f:
    f.write(df_summary[["model", "psnr", "ssim", "mae", "mse", "rmse", "lpips"]].to_latex(index=False, float_format="%.4f"))

model_rank_order = rank_df["model"].tolist()
image_best_scores = []
for fname, score_dict in per_image_scores.items():
    if not score_dict:
        continue
    valid_psnr = [(m, v["psnr"]) for m, v in score_dict.items() if pd.notna(v.get("psnr", np.nan))]
    if not valid_psnr:
        continue
    best_model = max(valid_psnr, key=lambda x: x[1])[0]
    best_psnr  = dict(valid_psnr)[best_model]
    worst_model = min(valid_psnr, key=lambda x: x[1])[0]
    worst_psnr  = dict(valid_psnr)[worst_model]
    image_best_scores.append({
        "image": fname,
        "best_model": best_model,
        "best_psnr": best_psnr,
        "worst_model": worst_model,
        "worst_psnr": worst_psnr,
        "spread": best_psnr - worst_psnr
    })

df_cases = pd.DataFrame(image_best_scores).sort_values(by="best_psnr", ascending=False).reset_index(drop=True)
df_hard  = pd.DataFrame(image_best_scores).sort_values(by="worst_psnr", ascending=True).reset_index(drop=True)

for fname in tqdm(gt_files, desc="Panels"):
    low = read_rgb(os.path.join(TEST_LOW_DIR, fname))
    gt  = read_rgb(os.path.join(GT_DIR, fname))
    if low is None or gt is None:
        continue

    preds_dict = {}
    metrics_dict = {}
    for model_name in model_rank_order:
        pred_path = os.path.join(MODEL_DIRS[model_name], fname)
        if os.path.exists(pred_path):
            pred = read_rgb(pred_path)
            if pred is None:
                continue
            pred = resize_to_gt(pred, gt)
            preds_dict[model_name] = pred
            if fname in per_image_scores and model_name in per_image_scores[fname]:
                metrics_dict[model_name] = per_image_scores[fname][model_name]

    if preds_dict:
        save_path = os.path.join(ALL_VIS_DIR, safe_filename(os.path.splitext(fname)[0]) + "_panel.png")
        make_panel(low, gt, preds_dict, metrics_dict, save_path, title=fname)

top_n = min(10, len(df_cases))
hard_n = min(10, len(df_hard))

for i in range(top_n):
    fname = df_cases.iloc[i]["image"]
    src = os.path.join(ALL_VIS_DIR, safe_filename(os.path.splitext(fname)[0]) + "_panel.png")
    if os.path.exists(src):
        Image.open(src).save(os.path.join(BEST_DIR, f"{i+1:02d}_{safe_filename(os.path.splitext(fname)[0])}.png"))

for i in range(hard_n):
    fname = df_hard.iloc[i]["image"]
    src = os.path.join(ALL_VIS_DIR, safe_filename(os.path.splitext(fname)[0]) + "_panel.png")
    if os.path.exists(src):
        Image.open(src).save(os.path.join(WORST_DIR, f"{i+1:02d}_{safe_filename(os.path.splitext(fname)[0])}.png"))

df_cases.to_csv(os.path.join(TABLE_DIR, "best_cases_by_psnr.csv"), index=False, encoding="utf-8-sig")
df_hard.to_csv(os.path.join(TABLE_DIR, "hard_cases_by_psnr.csv"), index=False, encoding="utf-8-sig")

print("\nDone.")
