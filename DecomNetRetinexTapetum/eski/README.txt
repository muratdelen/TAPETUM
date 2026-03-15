DecomNetRetinexTapetum project files

RGB attention kaldırıldı.
Sadece Tapetum attention kullanılır.

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
│   └── DecomNetRetinexTapetum/
└── DecomNetRetinexTapetum/
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── utils.py
    ├── train.py
    └── test.py

Colab:
%cd /content/TAPETUM/DecomNetRetinexTapetum
!python train.py
!python test.py