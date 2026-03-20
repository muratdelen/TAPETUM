RetinexTapetum project files

This standardized version follows the same training/reporting structure as the
reference Retinex + Tapetum code so outputs can be compared more fairly.

Main updates applied:
- Added detailed explanations/comments across all files.
- Switched lambda from softplus+clamp to sigmoid-bounded lambda:
      lambda = lambda_max * sigmoid(lambda_param)
- Added enhanced illumination smoothness loss on L_t (smooth_enh).
- Reduced attention regularization strength to be closer to the reference code.
- Added gradient clipping for more stable optimization.
- Fixed checkpoint history ordering so saved history includes the current epoch.
- Updated early stopping patience to a meaningful comparable value.
- Kept this variant single-attention only:
      RGB attention/gate is NOT used here.

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
│   └── RetinexTapetum/
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
