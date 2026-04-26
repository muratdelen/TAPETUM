# RetinexTapetum: Bio-Inspired Active Illumination Modeling for Efficient Low-Light Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)]
[![Dataset](https://img.shields.io/badge/Dataset-LOLv2-green.svg)]

---

## 🚀 Overview

Low-light image enhancement (LLIE) remains a challenging problem due to poor visibility, noise amplification, and structural degradation under insufficient illumination.

Retinex-based methods provide a physically grounded formulation by decomposing an image into reflectance and illumination components. However, most existing approaches treat illumination as a passive variable, which limits their effectiveness under complex and spatially non-uniform lighting conditions.

In this work, we propose **RetinexTapetum**, a bio-inspired LLIE framework that reformulates illumination modeling as an **active enhancement process**.

The method is inspired by the **Tapetum Lucidum**, which enhances vision in nocturnal animals by reusing photons under low-light conditions.

---

## 🧠 Method Formulation

### Retinex Decomposition

I(x) = R(x) · L(x)

### Tapetum Attention

T(x) = σ(f(L(x))) · (1 - L(x))

### Illumination Enhancement

L_t(x) = L(x) · (1 + λ T(x))

λ = λ_max · σ(λ_param)

### Reconstruction

I_enh(x) = R(x) · L_t(x)

---

## ⚙️ Pipeline

flowchart LR

A["Input"] --> B["Decomposition"]
B --> C["Reflectance R"]
B --> D["Illumination L"]

D --> E["Tapetum Attention"]
E --> F["T(x)"]

D --> G["Bounded λ"]
F --> H["L_t = L(1+λT)"]
G --> H

C --> I["Reconstruction"]
H --> I

I --> J["Enhanced Output"]

## 📊 Quantitative Results (LOLv2 Real Captured)

| Method | Params (M) ↓ | FPS ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| RetinexFormer | 1.6057 | 23.31 | 22.79 | 0.8386 | 0.1707 |
| RetinexTapetum (Ours) | 0.5247 | 85.03 | 19.25 | 0.7734 | 0.3669 |
| Zero-DCE | 0.0794 | 229.10 | 17.99 | 0.5778 | 0.3126 |
| RetinexNet | N/A | N/A | 15.95 | 0.6524 | 0.4128 |

---

## ⚡ Quick Start

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/TAPETUM"

python run_all_tapetum_models_colab.py

---

## 🔗 Code and Data

GitHub:
https://github.com/muratdelen/TAPETUM

Google Drive:
https://drive.google.com/drive/folders/13ayyEC3V1wWdX3AXdfL8y7VqnL8eTPFT

---

## 📖 Citation

@article{delen2026retinextapetum,
  title={RetinexTapetum: Bio-Inspired Active Illumination Modeling for Efficient Low-Light Image Enhancement},
  author={Delen, Murat and Ciftci, Serdar},
  year={2026}
}

---

## 👤 Authors

Murat Delen  
Harran University  

Serdar Ciftci  
Harran University  
