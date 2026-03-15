RetinexTapetum standardized project files

This version was updated to follow the same experimental standard used for the
other TAPETUM models so that outputs can be compared more fairly.

Main updates:
- detailed explanations added to all files
- sigmoid-bounded lambda instead of softplus-only lambda
- richer training loss: L1 + SSIM + color + attention + smooth_enh
- illumination_t exported in model output for better debugging
- train/val functions standardized
- early stopping added
- last.pth and best.pth checkpoint logic added
- history.csv export added
- test script aligned with training config

Folder contents:
- config.py
- dataset.py
- model.py
- losses.py
- utils.py
- train.py
- test.py

Expected relative structure:
TAPETUM/
├── datasets/
│   └── LoLv2/LOL-v2/Real_captured
├── LoLv2/
│   └── retinex-tapetum/
└── RetinexTapetum/
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── utils.py
    ├── train.py
    └── test.py

Colab:
%cd /content/TAPETUM/RetinexTapetum
!python train.py
!python test.py
