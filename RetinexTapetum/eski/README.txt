RetinexTapetumRGB project files

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
└── RetinexTapetumRGB/
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