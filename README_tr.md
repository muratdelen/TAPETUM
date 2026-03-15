
# TAPETUM: Biyolojik Esinli Düşük Işık Görüntü İyileştirme

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)]
[![Dataset](https://img.shields.io/badge/Dataset-LOLv2-green.svg)]
[![Colab](https://img.shields.io/badge/Run-Google%20Colab-orange.svg)]

Gece aktif hayvanlarda bulunan **tapetum lucidum** foton yansıtma mekanizmasından ilham alan biyolojik esinli düşük ışık görüntü iyileştirme framework’ü.

---

# İçindekiler
- Proje Özeti
- TAPETUM Mimarisi
- Genel Bakış
- Matematiksel Model
- Dataset
- Model Varyantları
- Nicel Sonuçlar
- Colab Hızlı Başlangıç
- Atıf

---

# Biyolojik Esinli Retinex Framework

TAPETUM, **Retinex ayrıştırmasını** ve **tapetum lucidum yansıma mekanizmasını** birleştiren biyolojik esinli bir düşük ışık görüntü iyileştirme çerçevesidir.

Amaç:
- karanlık sahnelerde aydınlatmayı artırmak
- reflectance yapısını korumak
- detay ve renk tutarlılığını muhafaza etmek

Repository dört ana model içerir:

- RetinexTapetum
- RetinexTapetumRGB
- DecomNetRetinexTapetum
- DecomNetRetinexTapetumRGB

---

# Genel Bakış

Birçok gece aktif hayvanın gözünde **tapetum lucidum** adı verilen yansıtıcı bir katman bulunur.

Bu katman:

- gelen ışığı retinaya geri yansıtır
- fotonların ikinci kez yakalanmasını sağlar
- düşük ışıkta görme kabiliyetini artırır

TAPETUM bu biyolojik mekanizmayı görüntü işleme algoritmasına dönüştürür.

---

# Matematiksel Model

## Klasik Retinex

I(x) = R(x) L(x)

I(x) : gözlenen görüntü  
R(x) : yansıma bileşeni  
L(x) : aydınlatma bileşeni  

## Retinex‑Tapetum

T(x) = σ(f(L(x))) (1-L(x))

Lt(x) = L(x)(1 + λT(x))

I_enh(x) = R(x) Lt(x)

## Retinex‑Tapetum RGB

Lc(x) = Lc(x)(1 + λTc(x))

c ∈ {R,G,B}

---

# Dataset

Deneyler **LOLv2 Real Captured dataset** üzerinde yapılmıştır.

datasets/
└── LoLv2/
    └── LOL-v2/
        └── Real_captured/
            ├── Train/
            │   ├── Low/
            │   └── Normal/
            └── Test/
                ├── Low/
                └── Normal/

---

# Model Varyantları

| Model | Açıklama |
|------|------|
RetinexTapetum | Retinex + Tapetum aydınlatma güçlendirme |
RetinexTapetumRGB | RGB kanal bazlı Tapetum |
DecomNetRetinexTapetum | DecomNet + Tapetum |
DecomNetRetinexTapetumRGB | DecomNet + RGB Tapetum |
RetinexNet | Karşılaştırma için temel model |

---

# Nicel Sonuçlar

| Model | PSNR | SSIM |
|------|------|------|
DecomNetRetinexTapetumRGB | 19.29 | 0.763 |
DecomNetRetinexTapetum | 19.24 | 0.773 |
RetinexNet | 15.95 | 0.652 |
RetinexTapetumRGB | 12.41 | 0.421 |
RetinexTapetum | 11.91 | 0.394 |

---

# Google Colab Hızlı Başlangıç

Projeyi çalıştırmanın en kolay yolu:

TAPETUM.ipynb

Adımlar:

1. Drive bağla
2. Projeyi /content/TAPETUM klasörüne kopyala
3. Modelleri eğit
4. Test et
5. Metrikleri hesapla
6. Sonuçları Drive’a kaydet

---

## Alıntı

Bu depoyu araştırmanızda kullanıyorsanız şu şekilde atıf yapabilirsiniz:

```bibtex
@article{delen2026tapetum,
  title={Tapetum-Retinex: A Bio-Inspired Retinex Framework for Low-Light Image Enhancement},
  author={Delen, Murat},
  year={2026}
}
```

---

# Yazar

Murat Delen  
Bilgisayar Mühendisliği  
Harran Üniversitesi  

GitHub: https://github.com/muratdelen

---

# Lisans

Bu repository araştırma ve akademik amaçlı kullanıma yöneliktir.
