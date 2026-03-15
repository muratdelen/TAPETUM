import os
import re
import csv
import sys
import json
import shutil
import subprocess
from pathlib import Path

# ============================================================
# RUN ALL TAPETUM MODELS (TRAIN + TEST + SUMMARY)
# Google Colab / local Python runner
#
# Beklenen klasör yapısı örneği:
# /content/TAPETUM/
#   RetinexTapetum/
#   RetinexTapetumRGB/
#   DecomNetRetinexTapetum/
#   DecomNetRetinexTapetumRGB/
#
# Her model klasörü içinde en az:
#   config.py
#   train.py
#   test.py
#
# Amaç:
# 1) İstenen modelleri sırayla eğitmek
# 2) Test scriptlerini çalıştırmak
# 3) Çıktılardan PSNR/SSIM/MAE/MSE gibi metrikleri çekmek
# 4) Tek bir CSV özet dosyası üretmek
# ============================================================

ROOT = "/content/TAPETUM"
SUMMARY_DIR = os.path.join(ROOT, "comparison_results")
os.makedirs(SUMMARY_DIR, exist_ok=True)

TARGET_EPOCHS = 120
RUN_TRAIN = True
RUN_TEST = True
STOP_ON_ERROR = False

MODELS = {
    "RetinexTapetum": os.path.join(ROOT, "RetinexTapetum"),
    "RetinexTapetumRGB": os.path.join(ROOT, "RetinexTapetumRGB"),
    "DecomNetRetinexTapetum": os.path.join(ROOT, "DecomNetRetinexTapetum"),
    "DecomNetRetinexTapetumRGB": os.path.join(ROOT, "DecomNetRetinexTapetumRGB"),
}

# ------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------

def print_header(title):
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def safe_read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def safe_write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def update_config_epochs(config_path, new_epochs):
    """
    config.py içinde EPOCHS = ... satırını günceller.
    Yoksa sona ekler.
    """
    if not os.path.exists(config_path):
        print("config.py bulunamadı:", config_path)
        return False

    text = safe_read(config_path)
    pattern = r"(?m)^EPOCHS\s*=\s*\d+\s*$"

    if re.search(pattern, text):
        text = re.sub(pattern, f"EPOCHS = {new_epochs}", text)
    else:
        text += f"\nEPOCHS = {new_epochs}\n"

    safe_write(config_path, text)
    print(f"[OK] EPOCHS -> {new_epochs} | {config_path}")
    return True


def ensure_history_csv_dir(model_dir):
    Path(model_dir).mkdir(parents=True, exist_ok=True)


def run_python_file(py_path, workdir):
    """Python dosyasını gerçek zamanlı çıktı akışı ile çalıştırır."""
    cmd = [sys.executable, py_path]
    print("[RUN]", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured = []
    for line in process.stdout:
        print(line, end="")
        captured.append(line)

    process.wait()
    return process.returncode, "".join(captured)


def find_metrics_csv(model_dir):
    """
    Test sonrası oluşmuş olabilecek csv dosyalarını arar.
    Öncelik: summary, metrics, test_metrics geçen dosyalar.
    """
    candidates = []
    for root, _, files in os.walk(model_dir):
        for name in files:
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(root, name)
            lower = name.lower()
            score = 0
            if "summary" in lower:
                score += 5
            if "test_metrics" in lower:
                score += 4
            if "metrics" in lower:
                score += 3
            if "history" in lower:
                score -= 10
            candidates.append((score, path))

    candidates.sort(reverse=True)
    return [p for _, p in candidates]


def try_parse_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def normalize_metric_name(name):
    n = name.strip().lower()
    aliases = {
        "psnr": "psnr",
        "avg_psnr": "psnr",
        "mean_psnr": "psnr",
        "ssim": "ssim",
        "avg_ssim": "ssim",
        "mean_ssim": "ssim",
        "mae": "mae",
        "avg_mae": "mae",
        "mean_mae": "mae",
        "mse": "mse",
        "avg_mse": "mse",
        "mean_mse": "mse",
        "lpips": "lpips",
        "avg_lpips": "lpips",
        "mean_lpips": "lpips",
    }
    return aliases.get(n, n)


def extract_metrics_from_csv(csv_path):
    """
    CSV biçimi değişebileceği için esnek parser.
    Aşağıdaki tipleri destekler:
    1) sütunlar direkt psnr/ssim/mae/mse içerir
    2) tek satır summary tablosu
    3) metric,value yapısı
    """
    metrics = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"[WARN] CSV okunamadı: {csv_path} | {e}")
        return metrics

    if not rows:
        return metrics

    # case 1: direct columns
    preferred = ["psnr", "ssim", "mae", "mse", "lpips"]
    for col in rows[0].keys():
        key = normalize_metric_name(col)
        if key in preferred:
            vals = [try_parse_float(r.get(col)) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                metrics[key] = sum(vals) / len(vals)

    # case 2: metric,value table
    if not metrics:
        cols = {c.strip().lower() for c in rows[0].keys()}
        if "metric" in cols and "value" in cols:
            metric_col = next(c for c in rows[0].keys() if c.strip().lower() == "metric")
            value_col = next(c for c in rows[0].keys() if c.strip().lower() == "value")
            for r in rows:
                k = normalize_metric_name(r[metric_col])
                v = try_parse_float(r[value_col])
                if v is not None:
                    metrics[k] = v

    return metrics


def extract_metrics_from_text(text):
    """CSV bulunamazsa log içinden metrik yakala."""
    patterns = {
        "psnr": [r"psnr\s*[:=]\s*([0-9]+\.?[0-9]*)", r"val_psnr\s*([0-9]+\.?[0-9]*)"],
        "ssim": [r"ssim\s*[:=]\s*([0-9]+\.?[0-9]*)"],
        "mae": [r"mae\s*[:=]\s*([0-9]+\.?[0-9]*)"],
        "mse": [r"mse\s*[:=]\s*([0-9]+\.?[0-9]*)"],
        "lpips": [r"lpips\s*[:=]\s*([0-9]+\.?[0-9]*)"],
    }
    result = {}
    lower = text.lower()
    for key, pats in patterns.items():
        found = []
        for pat in pats:
            found += re.findall(pat, lower)
        vals = [float(x) for x in found] if found else []
        if vals:
            result[key] = vals[-1]
    return result


def read_best_epoch_from_history(model_dir):
    history_files = []
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name.lower() == "history.csv":
                history_files.append(os.path.join(root, name))
    if not history_files:
        return None, None

    path = sorted(history_files)[0]
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None, path

    best_epoch = None
    best_psnr = None
    for r in rows:
        epoch = try_parse_float(r.get("epoch"))
        psnr = try_parse_float(r.get("val_psnr"))
        if epoch is None or psnr is None:
            continue
        epoch = int(epoch)
        if best_psnr is None or psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch

    return (best_epoch, best_psnr), path


def save_summary_csv(rows, out_csv):
    fieldnames = [
        "model",
        "train_status",
        "test_status",
        "best_epoch",
        "best_val_psnr",
        "psnr",
        "ssim",
        "mae",
        "mse",
        "lpips",
        "metrics_source",
        "train_log",
        "test_log",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_json(rows, out_json):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def print_summary(rows):
    print_header("FINAL SUMMARY")
    for row in rows:
        print(
            f"{row['model']}: train={row['train_status']} | test={row['test_status']} | "
            f"best_epoch={row['best_epoch']} | best_val_psnr={row['best_val_psnr']} | "
            f"psnr={row['psnr']} | ssim={row['ssim']} | mae={row['mae']} | mse={row['mse']}"
        )


# ------------------------------------------------------------
# Ana akış
# ------------------------------------------------------------

def main():
    summary_rows = []

    print_header("RUN ALL TAPETUM MODELS")
    print("ROOT          :", ROOT)
    print("TARGET_EPOCHS :", TARGET_EPOCHS)
    print("RUN_TRAIN     :", RUN_TRAIN)
    print("RUN_TEST      :", RUN_TEST)
    print("SUMMARY_DIR   :", SUMMARY_DIR)

    for model_name, model_dir in MODELS.items():
        print_header(f"MODEL: {model_name}")

        train_path = os.path.join(model_dir, "train.py")
        test_path = os.path.join(model_dir, "test.py")
        config_path = os.path.join(model_dir, "config.py")

        row = {
            "model": model_name,
            "train_status": "SKIPPED",
            "test_status": "SKIPPED",
            "best_epoch": "",
            "best_val_psnr": "",
            "psnr": "",
            "ssim": "",
            "mae": "",
            "mse": "",
            "lpips": "",
            "metrics_source": "",
            "train_log": "",
            "test_log": "",
        }

        if not os.path.isdir(model_dir):
            print("[ERROR] Model klasörü yok:", model_dir)
            row["train_status"] = "MISSING_DIR"
            row["test_status"] = "MISSING_DIR"
            summary_rows.append(row)
            if STOP_ON_ERROR:
                break
            continue

        model_summary_dir = os.path.join(SUMMARY_DIR, model_name)
        os.makedirs(model_summary_dir, exist_ok=True)

        # -------------------------
        # Train
        # -------------------------
        train_log_text = ""
        if RUN_TRAIN:
            if os.path.exists(config_path):
                update_config_epochs(config_path, TARGET_EPOCHS)
            else:
                print("[WARN] config.py bulunamadı, epoch güncellenemedi.")

            if os.path.exists(train_path):
                code, output = run_python_file(train_path, model_dir)
                train_log_path = os.path.join(model_summary_dir, "train_log.txt")
                safe_write(train_log_path, output)
                row["train_log"] = train_log_path
                train_log_text = output
                row["train_status"] = "OK" if code == 0 else f"ERROR_{code}"
                print(f"[TRAIN STATUS] {row['train_status']}")
                if code != 0 and STOP_ON_ERROR:
                    summary_rows.append(row)
                    break
            else:
                print("[ERROR] train.py bulunamadı:", train_path)
                row["train_status"] = "MISSING_TRAIN"
        
        # history.csv üzerinden best epoch/val_psnr çek
        best_info, history_path = read_best_epoch_from_history(model_dir)
        if best_info is not None:
            row["best_epoch"] = best_info[0]
            row["best_val_psnr"] = round(best_info[1], 6)
            print(f"[HISTORY] best_epoch={row['best_epoch']} | best_val_psnr={row['best_val_psnr']}")
            if history_path:
                shutil.copy2(history_path, os.path.join(model_summary_dir, "history.csv"))

        # -------------------------
        # Test
        # -------------------------
        test_log_text = ""
        if RUN_TEST:
            if os.path.exists(test_path):
                code, output = run_python_file(test_path, model_dir)
                test_log_path = os.path.join(model_summary_dir, "test_log.txt")
                safe_write(test_log_path, output)
                row["test_log"] = test_log_path
                test_log_text = output
                row["test_status"] = "OK" if code == 0 else f"ERROR_{code}"
                print(f"[TEST STATUS] {row['test_status']}")
            else:
                print("[ERROR] test.py bulunamadı:", test_path)
                row["test_status"] = "MISSING_TEST"

        # -------------------------
        # Metrik topla
        # -------------------------
        metrics = {}
        csv_candidates = find_metrics_csv(model_dir)
        if csv_candidates:
            for csv_path in csv_candidates:
                metrics = extract_metrics_from_csv(csv_path)
                if metrics:
                    row["metrics_source"] = csv_path
                    shutil.copy2(csv_path, os.path.join(model_summary_dir, os.path.basename(csv_path)))
                    break

        if not metrics:
            merged_text = train_log_text + "\n" + test_log_text
            metrics = extract_metrics_from_text(merged_text)
            if metrics:
                row["metrics_source"] = "parsed_from_logs"

        for key in ["psnr", "ssim", "mae", "mse", "lpips"]:
            if key in metrics:
                row[key] = round(metrics[key], 6)

        summary_rows.append(row)

    # Kaydet
    out_csv = os.path.join(SUMMARY_DIR, "all_models_summary.csv")
    out_json = os.path.join(SUMMARY_DIR, "all_models_summary.json")
    save_summary_csv(summary_rows, out_csv)
    save_summary_json(summary_rows, out_json)

    print_summary(summary_rows)
    print("\nCSV :", out_csv)
    print("JSON:", out_json)


if __name__ == "__main__":
    main()
