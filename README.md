# TAPETUM: Bio-Inspired Low-Light Image Enhancement

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)]
[![Dataset](https://img.shields.io/badge/Dataset-LOLv2-green.svg)]

TAPETUM is a **bio-inspired low-light image enhancement framework**  
motivated by the **tapetum lucidum** photon reflection mechanism in nocturnal animals.

---

## 🚀 Overview

Low-light image enhancement (LLIE) is fundamentally an **illumination recovery problem**.

Classical Retinex:
I(x) = R(x) · L(x)

TAPETUM:
L_t(x) = L(x)(1 + λT(x))  
I_enh(x) = R(x) · L_t(x)

👉 Key difference:

- Retinex → passive illumination correction  
- TAPETUM → **active and adaptive illumination amplification**

---

## 🧠 Core Idea

In biological systems (e.g., cats, deer):

- Light is reflected back through the retina  
- The same photons are reused  
- Visibility improves in low-light conditions  

Simplified model:
I_eff = I + rI

TAPETUM formulation:
T(x) = sigmoid(f(L(x))) · (1 - L(x))  
L_t(x) = L(x)(1 + λT(x))

---

## ⚙️ Pipeline Comparison

```mermaid
flowchart LR

subgraph Classical_Retinex
A1["Low-Light Input"] --> B1["Decomposition"]
B1 --> C1["Reflectance R"]
B1 --> D1["Illumination L"]
D1 --> E1["Global Illumination Adjustment"]
C1 --> F1["Reconstruction"]
E1 --> F1
F1 --> G1["Output"]
end

subgraph RetinexTapetum
A2["Low-Light Input"] --> B2["DecomNet"]
B2 --> C2["R_low"]
B2 --> D2["L_low"]
D2 --> E2["TapetumAttentionNet"]
E2 --> F2["T = sigmoid(raw)·(1-L)"]
D2 --> H2["λ = λ_max·sigmoid(λ_param)"]
F2 --> I2["L_t = L·(1+λT)"]
H2 --> I2
C2 --> J2["Reconstruction"]
I2 --> J2
J2 --> K2["Enhanced Output"]
end
```

---

## 🧩 Architecture

```mermaid
flowchart LR

A["Input I(x)"] --> B["Decomposition"]
B --> C["Reflectance R(x)"]
B --> D["Illumination L(x)"]

D --> E["Tapetum Attention"]
E --> F["Enhanced Illumination L_t(x)"]

C --> G["Reconstruction"]
F --> G

G --> H["Output"]
```

---

## 📊 Results (LOLv2 Real)

| Model | PSNR ↑ | SSIM ↑ |
|------|------:|------:|
| RetinexNet | 15.95 | 0.652 |
| **RetinexTapetum** | **19.24** | **0.773** |

---

## 📁 Dataset

LOLv2 Real Captured:

datasets/LoLv2/LOL-v2/Real_captured/

---

## ⚡ Quick Start

python run_all_tapetum_models_colab.py

---

## 📖 Citation

@article{delen2026tapetum,
  title={Tapetum-Retinex: A Bio-Inspired Low-Light Image Enhancement},
  author={Delen, Murat},
  year={2026}
}

---

## 👤 Author

Murat Delen  
Harran University  
Computer Engineering
