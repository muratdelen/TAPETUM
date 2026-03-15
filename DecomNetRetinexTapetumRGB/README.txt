DecomNetRetinexTapetumRGB project files - standardized version

Updated to match the reference training standard:
- sigmoid-bounded lambda
- stronger edge-aware smoothing on enhanced illumination (L_t)
- lighter attention/gate regularization weights
- standardized train/val log fields
- fixed checkpoint history ordering
- gradient clipping added
- early stopping aligned with reference setting

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
│   └── DecomNetRetinexTapetumRGB/
└── DecomNetRetinexTapetumRGB/
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── utils.py
    ├── train.py
    └── test.py

Colab:
%cd /content/TAPETUM/DecomNetRetinexTapetumRGB
!python train.py
!python test.py
