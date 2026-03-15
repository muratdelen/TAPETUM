
# TAPETUM: Biyolojik İlhamlı Düşük Işık Görüntü İyileştirme

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red.svg)]
[![Veri Kümesi](https://img.shields.io/badge/Dataset-LOLv2-green.svg)]
[![Colab](https://img.shields.io/badge/Run-Google%20Colab-orange.svg)]

Gece hayvanlarında bulunan **tapetum lucidum** foton yansıma mekanizmalarından esinlenilmiş, biyolojik olarak esinlenilmiş düşük ışıklı görüntü iyileştirme çerçevesi.

---

## İçindekiler

- Proje Öne Çıkanları
- TAPETUM Mimarlık
- Genel Bakış
- Çerçeve Mimarisi
- Matematiksel Formülasyon
- Biyolojik İlham
- Biyolojik Görüş → TAPETUM Algoritması
- Metriklerin Biyolojik Yorumlanması
- Nicel Sonuçlar
- İlgili Çalışmalar
- Colab Hızlı Başlangıç
- Eğitim ve Değerlendirme
- Alıntı

---

### Düşük Işıkta Görüntü İyileştirme için Biyolojik Esintili Retinex Çerçevesi

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DeepLearning-red">
<img alt="Dataset" src="https://img.shields.io/badge/Dataset-LOLv2-green">
<img alt="Task" src="https://img.shields.io/badge/Task-Low--Light%20Enhancement-orange">
<img alt="Lisans" src="https://img.shields.io/badge/License-Research-blue">
</p>

<p align="center"><b>Retinex + Tapetum Lucidum Esinli Aydınlatma Modellemesi</b></p>

TAPETUM, gece hayvanlarında gözlemlenen **Tapetum Lucidum'dan esinlenilmiş yansıma mekanizması** ile **Retinex ayrıştırmasını** birleştiren, biyolojik olarak esinlenilmiş düşük ışıklı görüntü iyileştirme çerçevesidir. Amaç, yansıma yapısını, mekansal ayrıntıyı ve renk tutarlılığını korurken karanlık sahnelerde aydınlatma geri kazanımını iyileştirmektir.

Bu depoda dört ana model varyantı bulunmaktadır:

- **RetinexTapetum**
- **RetinexTapetumRGB**
- **DecomNetRetinexTapetum**
- **DecomNetRetinexTapetumRGB**

---


---


---

## Proje Öne Çıkanları

TAPETUM çerçevesi, düşük ışık koşullarında görüntü iyileştirme için biyolojik olarak esinlenilmiş bir mekanizma sunmaktadır .
Projenin temel katkıları aşağıda özetlenmiştir.

deniz kızı
akış şeması LR

Biyolojik İlham
(Tapetum Lucidum)"]

A --> B["Foton Yansıma Mekanizması"]

B --> C["Aydınlatma Amplifikasyon Modeli"]

C --> D["Tapetum Dikkat Modülü"]

D --> E["Retinex / DecomNet Entegrasyonu"]

E --> F["Geliştirilmiş Düşük Işık Görüntüleri"]
```

### Başlıca Katkılar

- **Biyolojik İlhamlı Aydınlatma Güçlendirme**
*Tapetum lucidum* mekanizmasının hesaplamalı bir yorumunu sunar.

- **Tapetum Dikkat Modülü**
Öğrenilebilir dikkat mekanizması kullanarak aydınlatmayı artırır.

- **RGB Spektral Tapetum Varyantı**
Ren geyiklerinin mevsimsel spektral adaptasyonundan esinlenilmiştir.

- **Retinex Çerçeveleriyle Uyumluluk**
Hem klasik Retinex ayrıştırmasıyla hem de öğrenilmiş ayrıştırma ağlarıyla çalışır.

- **Geliştirilmiş Nicel Performans**
PSNR ve SSIM metriklerini kullanarak LOLv2 veri kümesinde güçlü performans sergiliyor.


---

## Yöntem Karşılaştırması: Retinex vs TAPETUM vs TAPETUM RGB

deniz kızı
akış şeması LR

alt grafik Klasik_Retinex
A["Düşük Işık Görüntüsü I(x)"]

A --> B["Retinex Ayrışması"]
B --> C["Yansıma R(x)"]
B --> D["Aydınlatma L(x)"]

D --> E["Aydınlatma Ayarı"]

C --> F["Yeniden Yapılanma"]
E --> F

F --> G["Geliştirilmiş Görüntü"]
son

alt grafik TAPETUM_Çerçevesi
H["Düşük Işık Görüntüsü I(x)"]

H --> I["Retinex / DecomNet"]
I --> J["Yansıma R(x)"]
I --> K["Aydınlatma L(x)"]

K --> L["Tapetum Dikkat T(x)"]

L --> M["Aydınlatma Amplifikasyonu Lt(x)=L(x)(1+λT(x))"]

J --> N["Yeniden Yapılanma"]
M --> N

N --> O["Geliştirilmiş Görüntü"]
son

alt grafik TAPETUM_RGB
P["Düşük Işık Görüntüsü I(x)"]

P --> Q["Ayrıştırma"]

Q --> R["Yansıma R(x)"]
Q --> S["Aydınlatma L(x)"]

S --> TR["Tapetum R"]
S --> TG["Tapetum G"]
S --> TB["Tapetum B"]

TR --> LR["L_R(x)(1+λT_R(x))"]
TG --> LG["L_G(x)(1+λT_G(x))"]
TB --> LB["L_B(x)(1+λT_B(x))"]

R --> U["Yeniden Yapılanma"]
LR --> U
LG --> U
LB --> U

U --> V["Geliştirilmiş RGB Görüntüsü"]
son
```


## TAPETUM Komple Mimari

deniz kızı
akış şeması LR

Düşük Işıkta Görüntü
I(x)"]

A --> B["Görüntü Ayrıştırması"]
Retinex / DecomNet"]

B --> C["Yansıma
R(x)"]

B --> D["Aydınlatma"
L(x)"]

D --> E["Tapetum Dikkat Modülü
T(x)=σ(f(L(x)))(1-L(x))"]

E --> F["Aydınlatma Amplifikasyonu"
Lt(x)=L(x)(1+λT(x))"]

C --> G["Görüntü Yeniden Yapılandırma"]

F --> G

G --> H["Geliştirilmiş Görüntü"
I_enh(x)=R(x)L(x)(1+λT(x))"]
```

### TAPETUM RGB Mimari

deniz kızı
akış şeması LR

A["Düşük Işıkta Görüntü"]

A --> B["Retinex / DecomNet Ayrıştırması"]

B --> C["Yansıma R"]

B --> D["Aydınlatma L"]

D --> E["Tapetum RGB Modülü"]

E --> ER["Tapetum R"]
E --> EG["Tapetum G"]
E --> EB["Tapetum B"]

ER --> LR["L_R(x)(1+λT_R(x))"]
EG --> LG["L_G(x)(1+λT_G(x))"]
EB --> LB["L_B(x)(1+λT_B(x))"]

C --> O["Yeniden Yapılanma"]

LR --> O
LG --> O
LB --> O

O --> P["Geliştirilmiş RGB Görüntüsü"]
```


## Genel Bakış

Gece aktif olan birçok hayvanın, gelen fotonları retinaya geri yansıtan ve düşük ışık koşullarında görmeyi iyileştiren **tapetum lucidum** adı verilen yansıtıcı bir göz tabakası vardır. TAPETUM, bu biyolojik fikri düşük ışıkta görüntü iyileştirme için hesaplamalı bir çerçeveye dönüştürüyor.

Temel fikir basit:

1. Giriş görüntüsünü **yansıma** ve **aydınlatma** bileşenlerine ayırın.
2. **Tapetumdan esinlenilmiş bir yansıtma modülü** kullanarak aydınlatmayı iyileştirin.
3. Son iyileştirilmiş görüntüyü yeniden oluşturun.

---

## Çerçeve Mimarisi

deniz kızı
akış şeması TD

A [Düşük Işıkta Görüntü] --> B [Retinex Ayrıştırması]

B --> C[Yansıma R]
B --> D[Aydınlatma L]

D --> E[Tapetum Dikkat Modülü]
E --> F[Gelişmiş Aydınlatma Işığı]

C --> G[Görüntü Yeniden Yapılandırma]
F --> G

G --> H [Geliştirilmiş Görüntü]
```

### Retinex-Tapetum

deniz kızı
akış şeması LR

A[Giriş Görüntüsü I] --> B[Retinex Ayrıştırması]

B --> C[Yansıma R]
B --> D[Aydınlatma L]

D --> E[Tapetum Dikkat Modülü]
E --> F[Gelişmiş Aydınlatma Işığı]

C --> G[Yeniden Yapılanma]
F --> G

G --> H [Geliştirilmiş Görüntü]
```

### Retinex-Tapetum-RGB

deniz kızı
akış şeması TD

A[Giriş Görüntüsü]

A --> B [Retinex Ayrışması]

B --> R[Yansıma R]
B --> L[Aydınlatma L]

L --> RGB[TapetumRGB]

RGB --> TR [Tapetum R Kanalı]
RGB --> TG [Tapetum G Kanalı]
RGB --> TB [Tapetum B Kanalı]

R --> O[Yeniden Yapılanma]

TR --> ER [Geliştirilmiş R]
TG --> EG [Geliştirilmiş G]
TB --> EB [Geliştirilmiş B]

ER --> O
EG --> O
EB --> O

O --> F[Geliştirilmiş RGB Görüntü]
```

### TAPETUM'un Tam Boru Hattı

deniz kızı
akış şeması TD

[Düşük Işıkta Çekilmiş Görüntü]

A --> B[DecomNet / Retinex]

B --> C[Yansıma R]
B --> D[Aydınlatma L]

C --> G[Yeniden Yapılanma]

D --> E[Tapetum Dikkat Modülü]
E --> F [Gelişmiş Aydınlatma]
F --> G

G --> H [Geliştirilmiş Görüntü]
```

---



## TAPETUM Çerçevesinin Tamamlanması

TAPETUM çerçevesi, Retinex ayrıştırmasını, Tapetum'dan ilham alan aydınlatma iyileştirmesini ve isteğe bağlı RGB kanal duyarlı modülasyonunu entegre eder.

deniz kızı
akış şeması TD

Düşük Işık Girişi
I(x)"] --> B["Ayrıştırma"]

B --> C1["Retinex
I(x)=R(x)L(x)"]
B --> C2["DecomNet
(R,L)=DecomNet(I)"]

C1 --> D1["RetinexTapetum"]
C1 --> D2["RetinexTapetumRGB"]

C2 --> D3["DecomNetRetinexTapetum"]
C2 --> D4["DecomNetRetinexTapetumRGB"]

D1 --> E1["Tapetum
T(x)=σ(f(L(x)))(1-L(x))"]
E1 --> F1["Lt(x)=L(x)(1+λT(x))"]
F1 --> G1["Ienh(x)=R(x)Lt(x)"]

D2 --> E2["Tbase(x)=σ(f(L(x))) ⊙ (1-L(x))"]
E2 --> F2["gc=1+s tanh(αc)"]
F2 --> G2["Trgb,c(x)=Tbase,c(x)gc"]
G2 --> H2["Lt,c(x)=Lc(x)(1+λTrgb,c(x))"]
H2 --> I2["Ienh,c(x)=Rc(x)Lt,c(x)"]

D3 --> E3["T(x)=σ(f(L(x)))(1-L(x))"]
E3 --> F3["Lt(x)=L(x)(1+λT(x))"]
F3 --> G3["Ienh(x)=R(x)Lt(x)"]

D4 --> E4["Tbase(x)=σ(f(L(x))) ⊙ (1-L(x))"]
E4 --> F4["gc=1+s tanh(αc)"]
F4 --> G4["Trgb,c(x)=Tbase,c(x)gc"]
G4 --> H4["Lt,c(x)=Lc(x)(1+λTrgb,c(x))"]
H4 --> I4["Ienh,c(x)=Rc(x)Lt,c(x)"]

G1 --> Z["Gelişmiş Çıkış"]
I2 --> Z
G3 --> Z
I4 --> Z
```

### Model Ailesine Genel Bakış

| Model | Ayrıştırma | Tapetum | RGB Modülasyonu |
|---|---|---|---|
| RetinexTapetum | Retinex | ✓ | ✗ |
| RetinexTapetumRGB | Retinex | ✓ | ✓ |
| DecomNetRetinexTapetum | DecomNet | ✓ | ✗ |
| DecomNetRetinexTapetumRGB | DecomNet | ✓ | ✓ |

### Temel TAPETUM Denklemi

```matematik
I_{enh}(x)=R(x)L(x)(1+\lambda T(x))
```


## Matematiksel Formülasyon

### Klasik Retinex Modeli

```matematik
I(x) = R(x)\cdot L(x)
```

Neresi

- I(x), gözlemlenen düşük ışıklı görüntüdür.
- R(x) yansıma bileşenidir.
- L(x) aydınlatma bileşenidir.

---

### Retinex-Tapetum

Tapetum dikkat haritası:

```matematik
T(x) = \sigma(f(L(x)))\,(1-L(x))
```

Geliştirilmiş aydınlatma:

```matematik
L_t(x) = L(x)\,(1+\lambda T(x))
```

Yeniden yapılanma:

```matematik
I_{enh}(x) = R(x)\cdot L_t(x)
```

Kompakt formül:

```matematik
I_{enh}(x) = R(x)\cdot L(x)\,(1+\lambda T(x))
```

---

### Retinex-Tapetum-RGB

Temel dikkat:

```matematik
T_{base}(x) = \sigma(f(L(x)))\odot (1-L(x))
```

Kanal modülasyonu:

```matematik
g_c = 1 + s\tanh(\alpha_c), \quad c \in \{R,G,B\}
```

Kanala özgü Tapetum haritası:

```matematik
T^{rgb}_c(x) = T_{base,c}(x)\cdot g_c
```

Kanal başına geliştirilmiş aydınlatma:

```matematik
L_t^c(x) = L^c(x)\,(1+\lambda T^{rgb}_c(x)), \quad c \in \{R,G,B\}
```

Yeniden yapılanma:

```matematik
I_{enh}^c(x) = R^c(x)\cdot L_t^c(x)
```

Vektör biçimi:

```matematik
I_{enh}(x) = R(x)\odot L(x)\odot (1+\lambda T_{rgb}(x))
```


## Başlıca Katkılar

- **Biyolojik esinlenmeli aydınlatma modellemesi:** Tapetum lucidum mekanizmasına dayanmaktadır.
- **Tapetum yansıma modülü**, Retinex tabanlı bir iyileştirme işlem hattına entegre edilmiştir.
- **RGB kanalına duyarlı yansıma kontrolü**, ren geyiği gibi hayvanlardaki dalga boyu adaptasyonundan esinlenilmiştir.
- Daha güçlü yansıma ve aydınlatma ayrımı için **DecomNet tabanlı öğrenilmiş ayrıştırma**.
- **LOLv2 Gerçek Görüntüleri Üzerine Kapsamlı Değerlendirme:** Görsel ve nicel karşılaştırmalar kullanılarak yapılmıştır.

### Katkı Diyagramı

deniz kızı
akış şeması LR

A[Klasik Retinex Yöntemleri] --> B[RetinexNet]
A --> C[KinD / KinD++]

B --> D [Yansıma + Aydınlatma Modellemesi]
C --> D

D --> E [Aydınlatma Geliştirme]
E --> F [Önerilen TAPETUM Çerçevesi]

F --> G[Tapetum Lucidum İlham Veren Yansıma]
F --> H [Kanal Duyarlı RGB Yansıtma]
F --> [DecomNet ile Ayrıştırmayı Öğrendim]

G --> J [Geliştirilmiş Aydınlatma Geri Kazanımı]
H --> J
Ben --> J

J --> K [Geliştirilmiş Düşük Işık Görüntüleri]
```

### TAPETUM vs RetinexNet

deniz kızı
akış şeması LR

[Düşük Işıkta Çekilmiş Görüntü]

A --> B[RetinexNet]
B --> C[Yansıma R]
B --> D[Aydınlatma L]
D --> E [Aydınlatma Geliştirme]
C --> F[Yeniden Yapılanma]
E --> F
F --> G[Geliştirilmiş Görüntü]

A --> H[TAPETUM Çerçevesi]
H --> I[Retinex / DecomNet Ayrıştırması]
Ben --> J[Yansıma R]
Ben --> K[Aydınlatma L]
K --> L[Tapetum Dikkat Modülü]
L --> M [Gelişmiş Aydınlatma Işığı]
J --> N[Yeniden Yapılanma]
M --> N
N --> O[Geliştirilmiş Görüntü]
```

---

## Depo Yapısı

```metin
TAPETUM/
├ ── DecomNetRetinexTapetum/
├ ── DecomNetRetinexTapetumRGB/
├ ── LoLv2/
├ ── Metrikler/
├ ── RetinexTapetumRGB/
├ ── veri kümeleri/
├ ── retinex-tapetum/
└── README.md
```

---

## Veri Kümesi

Deneyler **LOLv2 Gerçek Yakalanmış Veri Kümesi** üzerinde gerçekleştirilmiştir.

### GitHub örnekleri
- `datasets/LoLv2/LOL-v2/Real_captured`
- Depo yolu: `https://github.com/muratdelen/TAPETUM/tree/main/datasets/LoLv2/LOL-v2/Real_captured`

### Google Drive veri seti
- **VERİ KÜMESİ İNDİRME**
`https://drive.google.com/drive/folders/1QO2_buG32OjDI2w3Cg1_8e5MquEww6Ix?usp=sharing`

Veri kümesinin düzeni:

```metin
veri kümeleri/
└── LoLv2/
└── LOL-v2/
└── Gerçek_çekim/
            ├ ── Tren/
│ ├ ── Düşük/
│ └── Normal/
└── Test/
                ├ ── Düşük/
└── Normal/
```

---

## Model Varyantları

| Model | Açıklama |
|---|---|
| **RetinexTapetum** | Tapetum'dan ilham alan aydınlatma yansımasıyla Retinex'in ayrışması |
| **RetinexTapetumRGB** | Kanal duyarlı RGB Tapetum yansıması |
| **DecomNetRetinexTapetum** | Öğrenilmiş ayrıştırma + Tapetum yansıması |
| **DecomNetRetinexTapetumRGB** | Öğrenilmiş ayrıştırma + RGB Tapetum yansıması |
| **RetinexNet** | Temel karşılaştırma modeli |

---

## Görsel Sonuçlar

### En iyi senaryoya göre niteliksel karşılaştırmalar

Bu arşivde aşağıdaki konularda özenle seçilmiş görsel karşılaştırmalar yer almaktadır:

- GitHub: `https://github.com/muratdelen/TAPETUM/tree/main/Metrics/visuals/best_cases'
- Google Drive sonuçları: `https://drive.google.com/drive/folders/1dTq0xWTz0xJL2ngVaFqajoVVtfNE2VgY?usp=sharing`

Bu dosyalarda şu gibi güçlü örnekler yer almaktadır:

- `01_00755.png`
- `02_00756.png`
- `03_00744.png`
- `04_00751.png`
- `05_00720.png`
- `06_00741.png`
- `07_00721.png`
- `08_00748.png`
- `09_00747.png`
- `10_00750.png`

### Örnek görsel karşılaştırmalar

<p align="center">
<img src="Metrics/visuals/best_cases/01_00755.png" width="900" alt="En iyi durum 00755">
</p>

<p align="center">
<img src="Metrics/visuals/best_cases/05_00720.png" width="900" alt="En iyi durum 00720">
</p>

<p align="center">
<img src="Metrics/visuals/best_cases/09_00747.png" width="900" alt="En iyi durum 00747">
</p>

### Modele özel çıktı klasörleri

#### GitHub sonuç klasörleri
- RetinexNet: 'https://github.com/muratdelen/TAPETUM/tree/main/LoLv2/RetinexNet/results/Test'
- RetinexTapetum: 'https://github.com/muratdelen/TAPETUM/tree/main/LoLv2/retinex-tapetum/results/Test'
- RetinexTapetumRGB: 'https://github.com/muratdelen/TAPETUM/tree/main/LoLv2/RetinexTapetumRGB/results/Test'
- DecomNetRetinexTapetum: 'https://github.com/muratdelen/TAPETUM/tree/main/LoLv2/DecomNetRetinexTapetum/results/Test'
- DecomNetRetinexTapetumRGB: `https://github.com/muratdelen/TAPETUM/tree/main/LoLv2/DecomNetRetinexTapetumRGB/results/Test`

#### Google Drive kaynakları
- **TAPETUM İNDİRME**
`https://drive.google.com/drive/folders/1EtT9abcdGIWMrzZ2zUGHB0A_gg7LMM8J?usp=sharing`
- **RETINEXNET İNDİRME**
`https://drive.google.com/drive/folders/1CKqjhcsQ5Fs8Btkn4jFoFXqCy9gZlh35?usp=sharing`
- **LOLV2 İNDİRME SONUCU**
`https://drive.google.com/drive/folders/1dTq0xWTz0xJL2ngVaFqajoVVtfNE2VgY?usp=sharing`

### Nitel gözlemler

En iyi örnek üzerinden yapılan görsel karşılaştırmalarda şu örüntüler gözlemlenmektedir:

- DecomNet tabanlı TAPETUM varyantları daha koyu bölgeleri daha etkili bir şekilde kurtarır.
- RGB sürümü genellikle renk dengesini ve spektral tutarlılığı iyileştirir.
- RetinexTapetum ve RetinexTapetumRGB, yöntem fikrini koruyor ancak nicel performansları DecomNet tabanlı varyantların altında kalıyor.
- DecomNetRetinexTapetumRGB, parlaklık, detay ve renk doğruluğu açısından genellikle en dengeli görsel sonucu üretir.

---



---

## Yönteme Genel Bakış

TAPETUM çerçevesi, biyolojik olarak esinlenilmiş bir aydınlatma yansıtma mekanizması ekleyerek klasik Retinex tabanlı düşük ışık iyileştirmesini genişletir. Model ailesi şunları entegre eder:

- Retinex tabanlı aydınlatma modellemesi
- Tapetumdan ilham alan düşünceli dikkat
- İsteğe bağlı RGB kanal duyarlı modülasyon
- DecomNet kullanılarak isteğe bağlı öğrenilmiş ayrıştırma

### Metot Akışı

deniz kızı
akış şeması LR

A["Düşük Işık Görüntüsü I(x)"] --> B["Retinex / DecomNet Ayrıştırması"]
B --> C["Yansıma R(x)"]
B --> D["Aydınlatma L(x)"]

D --> E["Tapetum Dikkat
T(x)=σ(f(L(x)))(1-L(x))"]

E --> F["Gelişmiş Aydınlatma"]
Lt(x)=L(x)(1+λT(x))"]

C --> G["Yeniden Yapılanma"]
F --> G

G --> H["Geliştirilmiş Görüntü"
Ienh(x)=R(x)L(x)(1+λT(x))"]
```

---



### Kağıt Tarzı Yöntem Şekli

deniz kızı
akış şeması LR

Düşük Işıkta Görüntü
I(x)"]

A --> B["Retinex / DecomNet
Ayrışma"]

B --> C["Yansıma
R(x)"]

B --> D["Aydınlatma"
L(x)"]

D --> E["Tapetum Dikkat
T(x)=σ(f(L(x)))(1-L(x))"]

E --> F["Aydınlatma Geliştirme"
Lt(x)=L(x)(1+λT(x))"]

C --> G["Yeniden Yapılanma"]

F --> G

G --> H["Geliştirilmiş Görüntü"
Ienh(x)=R(x)L(x)(1+λT(x))"]
```

### Model Varyantlarına Genel Bakış

deniz kızı
akış şeması TD

A["Düşük Işıkta Görüntü"]

A --> B["RetinexTapetum"]
A --> C["RetinexTapetumRGB"]
A --> D["DecomNetRetinexTapetum"]
A --> E["DecomNetRetinexTapetumRGB"]

B --> F["Geliştirilmiş Görüntü"]
C --> F
D --> F
E --> F
```

### Model Aile Tablosu

| Model | Ayrıştırma | Tapetum Dikkat | RGB Modülasyonu |
|---|---|---|---|
| RetinexTapetum | Retinex | ✓ | ✗ |
| RetinexTapetumRGB | Retinex | ✓ | ✓ |
| DecomNetRetinexTapetum | DecomNet | ✓ | ✗ |
| DecomNetRetinexTapetumRGB | DecomNet | ✓ | ✓ |

### TAPETUM Temel Denklemleri

```matematik
I_{enh}(x)=R(x)L(x)(1+\lambda T(x))
```

```matematik
I_{enh}(x)=R(x)\odot L(x)\odot(1+\lambda T_{rgb}(x))
```

### Karşılaştırma Görselleştirme Yapısı

```markdown
### Karşılaştırma Görselleştirme Yapısı

| Düşük Işık Girişi | Gerçek Değer | RetinexNet | RetinexTapetum | TapetumRGB | DecomNetTapetum | DecomNetTapetumRGB |
|------|------|------|------|------|------|------|
| ![](https://raw.githubusercontent.com/muratdelen/TAPETUM/main/datasets/LoLv2/LOL-v2/Real_captured/Test/Low/00750.png) | ![](https://raw.githubusercontent.com/muratdelen/TAPETUM/main/datasets/LoLv2/LOL-v2/Real_captured/Test/Normal/00750.png) | ![](https://raw.githubusercontent.com/muratdelen/TAPETUM/main/LoLv2/RetinexNet/results/Test/00750_S.png) | ![](https://raw.githubusercontent.com/muratdelen/TAPETUM/main/LoLv2/retinex-tapetum/results/Test/00750.png) | ![](https://raw.githubusercontent.com/muratdelen/TAPETUM/main/LoLv2/RetinexTapetumRGB/results/Test/00750.png) | ![](https://raw.githubusercontent.com/muratdelen/TAPETUM/main/LoLv2/DecomNetRetinexTapetum/results/Test/00750.png) | ![](https://raw.githubusercontent.com/muratdelen/TAPETUM/main/LoLv2/DecomNetRetinexTapetumRGB/results/Test/00750.png) |
```


## Eğitim Süreci

deniz kızı
akış şeması TD

Bir["Veri kümesi
LOLv2 Gerçek Yakalanan"] --> B["Eğitim Veri Yükleyici"]

B --> C["İleri Pas"]
Retinex / DecomNet"]

C --> D["Tapetum Modülü"]

D --> E["Yeniden Yapılanma"]

E --> F["Kayıp Hesaplaması
Yeniden Yapılandırma + Aydınlatma Düzenlemesi"]

F --> G["Geri yayılım"]

G --> H["Model Güncellemesi"]
```

---

## Sonuçlar Galerisi

Retinex bazlı yöntemler ve TAPETUM ailesini karşılaştıran örnek nitel sonuçlar.

deniz kızı
akış şeması LR

A["Düşük Işık Girişi"] --> B["RetinexNet"]
A --> C["RetinexTapetum"]
A --> D["RetinexTapetumRGB"]
A --> E["DecomNetRetinexTapetum"]
A --> F["DecomNetRetinexTapetumRGB"]

B --> G["Geliştirilmiş Çıktı"]
C --> G
D --> G
E --> G
F --> G
```

Bu karşılaştırmalar, TAPETUM çerçevesinin aydınlatma geri kazanımı ve renk tutarlılığında sağladığı iyileşmeyi göstermektedir.


## Nicel Sonuçlar

Aşağıdaki ortalama sonuçlar, veri deposundaki ölçüm tablolarından elde edilmiştir.

### Özet ölçümler

| Model | Eşleşen Dosyalar | PSNR ↑ | SSIM ↑ | MAE ↓ | MSE ↓ | RMSE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DecomNetRetinexTapetumRGB** | 100 | **19.2938** | 0.7632 | 24.6575 | 1009.2340 | 29.8147 | 0.3983 |
| **DecomNetRetinexTapetum** | 100 | 19.2473 | **0,7734** | 24.7627 | 997.9153 | 29.7785 | 0.3669 |
| RetinexNet | 100 | 15.9504 | 0.6524 | 0.1396 | 0.0284 | 0.1639 | Yok |
| RetinexTapetumRGB | 100 | 12.4179 | 0.4208 | 62.0526 | 4733.0982 | 65.0186 | **0,3411** |
| RetinexTapetum | 100 | 11.9131 | 0.3942 | 64.8876 | 5118.1268 | 68.1592 | 0.3541 |

### Sıralama özeti

| Model | Genel Sıralama | PSNR Sıralaması | SSIM Sıralaması | MAE Sıralaması | MSE Sıralaması | RMSE Sıralaması | LPIPS Sıralaması |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DecomNetRetinexTapetum** | **13,0** | 2 | 1 | 3 | 2 | 2 | 3 |
| **RetinexNet** | **13,0** | 3 | 3 | 1 | 1 | 1 | 4 |
| DecomNetRetinexTapetumRGB | 15.0 | 1 | 2 | 2 | 3 | 3 | 4 |
| RetinexTapetumRGB | 21.0 | 4 | 4 | 4 | 4 | 4 | 1 |
| RetinexTapetum | 27.0 | 5 | 5 | 5 | 5 | 5 | 2 |

### Görüntü başına kazanan sayısı

| Model | En İyi PSNR | En İyi SSIM | En İyi MAE | En İyi MSE | En İyi RMSE | En İyi LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| **DecomNetRetinexTapetum** | **39** | **69** | 0 | 0 | 0 | 44 |
| DecomNetRetinexTapetumRGB | 38 | 19 | 0 | 0 | 0 | 4 |
| RetinexNet | 15 | 3 | **100** | **100** | **100** | 0 |
| RetinexTapetumRGB | 8 | 9 | 0 | 0 | 0 | **52** |
| RetinexTapetum | 0 | 0 | 0 | 0 | 0 | 0 |

### Tercüme

- **DecomNetRetinexTapetumRGB**, en iyi ortalama **PSNR** değerini elde eder.
- **DecomNetRetinexTapetum**, hem **PSNR** hem de **SSIM**'de en iyi ortalama **SSIM** değerine ve görüntü başına en yüksek kazanma sayısına ulaşır.
- **RetinexNet**, diğer modellere kıyasla alışılmadık derecede düşük MAE/MSE/RMSE değerleri göstermektedir; bu da söz konusu hata ölçütlerinin farklı bir çıktı ölçeğinde veya dışa aktarma biçiminde olabileceğini düşündürmektedir. Bu değerler dikkatlice yorumlanmalıdır.
- Uygulamada, TAPETUM ailesinin en güçlü genel sonuçları **DecomNet tabanlı varyantlardan** elde edilmektedir.

### Metrik kaynakları

- GitHub metrik tabloları: `https://github.com/muratdelen/TAPETUM/tree/main/Metrics/tables`
- GitHub metrik görselleştirmeleri: `https://github.com/muratdelen/TAPETUM/tree/main/Metrics`
- Google Drive ölçümleri: `https://drive.google.com/drive/folders/13XOBg-1gWTgSrbhDkDteI1pIqVIdjCfE?usp=sharing`

---

## Kıyaslama Karşılaştırması

| Yöntem | PSNR ↑ | SSIM ↑ | Tip |
|---|---:|---:|---|
| RetinexNet | 15.95 | 0.652 | Retinex tabanlı derin öğrenme modeli |
| RetinexTapetum | 11.91 | 0,394 | Biyo-ilhamlı Retinex |
| RetinexTapetumRGB | 12.42 | 0.421 | Kanal duyarlı biyolojik ilhamlı Retinex |
| DecomNetRetinexTapetum | 19.25 | **0.773** | Retinex + Tapetum öğrenildi |
| **DecomNetRetinexTapetumRGB (TAPETUM)** | **19.29** | 0.763 | Tam TAPETUM modeli |

---


---


---


---

## TAPETUM'un Biyolojik İlhamı

---

## Biyolojik Görme → TAPETUM Algoritması

TAPETUM çerçevesi, gece hayvanlarında gözlemlenen biyolojik foton yansıma mekanizmalarından esinlenmiştir.
Aşağıdaki kavramsal şema, biyolojik fikrin hesaplamalı modele nasıl dönüştüğünü göstermektedir.

deniz kızı
akış şeması LR

A["Tapetum Lucidum
(Biyolojik Yansıtıcı Katman)"]

A --> B["Foton Yansıması"]

B --> C["İkinci Foton Yakalama"
Geliştirilmiş Işık Hassasiyeti"]

C --> D["Spektral Adaptasyon"
(Ren geyiği örneği)"]

D --> E["Aydınlatma Amplifikasyon Kavramı"]

E --> F["Tapetum Dikkat Modülü"]

F --> G["TAPETUM Görüntü İyileştirme Algoritması"]
```


TAPETUM çerçevesi, gece aktif olan hayvanlarda bulunan biyolojik görme mekanizmalarından esinlenmiştir. Düşük ışıklı ortamlarda aktif olan birçok hayvanın gözlerinde **tapetum lucidum** adı verilen yansıtıcı bir tabaka bulunur. Bu yapı, gelen ışığı retinadan geri yansıtarak, fotoreseptörlerin ilk geçişte emilmeyen fotonları yakalamasına olanak tanır.

Bu optik geri besleme mekanizması, karanlık koşullar altında kullanılabilir aydınlatmayı etkili bir şekilde artırır.

### Tapetum Yansıma Kavramı

Basitleştirilmiş bir modelde, retinaya ulaşan etkili ışık şu şekilde yaklaşık olarak ifade edilebilir:

```matematik
Etkin I = I + rI
```

Neresi:

- \(I\) gelen ışığı temsil eder.
- \(rI\), tapetum tabakasından yansıyan bileşeni temsil eder.

Bu durum algılanan parlaklığın artmasına yol açar ancak saçılma nedeniyle hafif bir mekansal yayılmaya neden olabilir.

### TAPETUM Algoritması İçin İlham Kaynağı

TAPETUM modeli bu kavramı bir görüntü iyileştirme işlem hattına dönüştürüyor:

1. **Retinex ayrıştırması** yansımayı ve aydınlatmayı birbirinden ayırır.
2. **Tapetum dikkati** daha karanlık bölgelerdeki aydınlatmayı artırır.
3. **Yeniden yapılandırma**, iyileştirilmiş görünürlüğe sahip gelişmiş görüntüyü üretir.

deniz kızı
akış şeması LR

A["Düşük Işıklı Görüntü"] --> B["Retinex / DecomNet Ayrıştırması"]
B --> C["Yansıma"]
B --> D["Aydınlatma"]

D --> E["Tapetum Dikkat Modülü"]
E --> F["Gelişmiş Aydınlatma"]

C --> G["Yeniden Yapılanma"]
F --> G

G --> H["Geliştirilmiş Görüntü"]
```

### Biyolojik Motivasyon

TAPETUM'un tasarım felsefesi, biyolojik sistemlerde gözlemlenen benzer bir denge ilkesini takip eder:

- Artan foton yakalama, **parlaklık algısını** iyileştirir.
- Spektral veya yansıtıcı amplifikasyon, **yapısal doğruluğu** hafifçe etkileyebilir.

**Aydınlatma geri kazanımı** ve **yapısal koruma** arasındaki bu denge, TAPETUM model varyantlarının performans özelliklerine yansımaktadır.


## Metrik Farklılıkların Biyolojik Yorumu

**DecomNetRetinexTapetumRGB** ve **DecomNetRetinexTapetum** arasındaki performans farklılıkları, TAPETUM çerçevesinin ardındaki biyolojik ilham kaynağı üzerinden yorumlanabilir.

Gece aktif olan birçok hayvanda **tapetum lucidum** olarak bilinen yansıtıcı bir retina tabakası bulunur. Bu yapı, gelen fotonları fotoreseptörlere doğru geri yansıtarak, düşük ışık koşullarında yakalanan ışık miktarını etkili bir şekilde artırır.

Basit bir ifadeyle, retinaya ulaşan etkin foton enerjisi şu şekilde tanımlanabilir:

```matematik
Etkin I = I + rI
```

burada \(r\) yansıyan ışık bileşenini temsil eder.

Bu mekanizma parlaklık algısını iyileştirir ancak yansıtıcı yüzeyden kaynaklanan saçılma nedeniyle hafif yapısal bozulmalara yol açabilir.

### Ren Geyiği Spektral Adaptasyonu

Ren geyiği gibi bazı hayvanlarda tapetum lucidum'un spektral davranışında mevsimsel değişiklikler görülür.

Kış aylarında:

- Tapetum, **mavi dalga boyu yansımasına** doğru kayar.
- Daha kısa dalga boyları daha güçlü saçılır.
- Loş ortamlarda daha fazla foton yakalanır.

Sonuç olarak:

- **Algılanan parlaklık artar**
- ancak **görsel yapıda hafif bir bozulma olabilir**

**Işık hassasiyeti** ve **yapısal bütünlük** arasındaki bu biyolojik denge, TAPETUM modellerinde gözlemlenen davranışa benzer.

### TAPETUM Metriklerinin Yorumlanması

Deneysel sonuçlarda:

| Model | PSNR | SSIM |
|---|---|---|
| DecomNetRetinexTapetumRGB | Daha yüksek | Biraz daha düşük |
| DecomNetRetinexTapetum | Biraz daha düşük | Daha yüksek |

#### DecomNetRetinexTapetumRGB

RGB varyantı, aydınlatmayı renk kanalları arasında bağımsız olarak artırır:

```matematik
L_t^c(x) = L^c(x)(1+\lambda T^{rgb}_c(x))
```

Bu daha güçlü spektral aydınlatma yükseltmesi, piksel düzeyinde yeniden yapılandırma doğruluğunu artırabilir ve bu da şunlara yol açar:

- daha yüksek **PSNR**

Ancak, kanal bazlı modülasyon, yapısal benzerliği azaltabilecek hafif spektral bozulmalara yol açabilir:

- biraz daha düşük **SSIM**

#### DecomNetRetinexTapetum

RGB olmayan varyant, tek bir aydınlatma geliştirme haritası uygular:

```matematik
L_t(x) = L(x)(1+\lambda T(x))
```

Bu, daha tutarlı mekansal aydınlatma sağlar ve kenarları ve dokuları daha etkili bir şekilde koruma eğilimindedir, sonuç olarak:

- daha yüksek **SSIM**

ancak piksel düzeyinde yeniden yapılandırma doğruluğu biraz daha düşük.

### Özet

Gözlemlenen metrik denge, TAPETUM çerçevesinin ardındaki biyolojik ilhamla tutarlıdır:

- **RGB Tapetum (spektral yükseltme)** → daha güçlü parlaklık geri kazanımı → daha yüksek PSNR
- **Standart Tapetum (yapı koruyucu yansıma)** → daha kararlı uzamsal yapı → daha yüksek SSIM

Bu davranış, spektral adaptasyonun yapısal hassasiyet pahasına ışık duyarlılığını artırdığı ren geyiği gibi hayvanlarda görülen biyolojik dengeyi yansıtmaktadır.


## Google Colab Hızlı Başlangıç

[![Colab'da Aç](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/muratdelen/TAPETUM/blob/main/TAPETUM.ipynb)


TAPETUM projesini çalıştırmanın en kolay yolu, sağlanan Colab not defterini kullanmaktır:

- Not Defteri: `TAPETUM.ipynb`
- GitHub Colab bağlantısı: `[GitHub'da TAPETUM.ipynb dosyasını açın](https://github.com/muratdelen/TAPETUM/blob/main/TAPETUM.ipynb)`

### Colab'da Aç

`TAPETUM.ipynb` dosyasının üst kısmındaki not defteri simgesini kullanın veya not defterini doğrudan Google Colab'deki depodan açın.

### Colab not defterinin işlevi

Bu not defteri, eksiksiz bir proje yürütme planı şeklinde düzenlenmiştir ve aşağıdaki aşamaları içermektedir:

1. **Colab'a kopyala**
Google Drive'ı bağlar ve proje klasörünü kopyalar.
`/content/drive/MyDrive/TAPETUM`
ile
`/content/TAPETUM`

2. **Tüm TAPETUM modellerini çalıştırın**
Proje düzeyindeki çalıştırıcıyı yürütür:

```bash
python /content/TAPETUM/run_all_tapetum_models_colab.py
```

3. **Tüm TAPETUM modellerini eğitin**
Aşağıdakiler için eğitim sürecini yürütür:
- `RetinexTapetum`
- `RetinexTapetumRGB`
- `DecomNetRetinexTapetum`
- `DecomNetRetinexTapetumRGB`

4. **Tüm TAPETUM modellerini test edin**
İlgili test komut dosyalarını çalıştırır ve model çıktılarını dışa aktarır.

5. **Tüm modelleri değerlendirin**
Aşağıdaki komut dosyası altında metrik karşılaştırma betiğini çalıştırır:

```bash
/content/TAPETUM/Metrics/evaluate_all_models_updated.py
```

6. **Sonuçları Google Drive'a geri senkronize edin**
Güncellenmiş TAPETUM çalışma alanını Drive'a geri kopyalar.

7. **İsteğe bağlı RetinexNet bazal değeri**
Defterde ayrıca şu konular için ayrı hücreler de bulunmaktadır:
- `RETINEXNET TRAIN`
- `RETINEXNET TEST`

### Önerilen uygulama sırası

Colab'da sorunsuz ve tam bir çalıştırma için şu sırayı kullanın:

1. **TAPETUM Driverdan yükle**
2. **tüm kodu oluştur**
3. **TÜM TAPETUM MODELLERİNİ EĞİTİN**
4. **TÜM TAPETUM MODELLERİNİ TEST EDİN**
5. **evaluate_all_models_updated.py çalıştırır**
6. **TAPETUM → DRIVE SENKRON KAYIT**

Temel karşılaştırmayı da dahil etmek istiyorsanız, şunu da çalıştırın:

7. **RETINEXNET TRENİ**
8. **RETINEXNET TESTİ**

### Drive'da Gerekli Klasör Yapısı

Not defterini çalıştırmadan önce, Google Drive'da aşağıdaki klasörün mevcut olduğundan emin olun:

```metin
Sürücüm/
└── TAPETUM/
```

Colab not defteri, projenin şu konuma kopyalanmasını bekliyor:

```metin
/içerik/TAPETUM
```

### Pratik notlar

- Bu not defteri **Google Colab'da çalıştırılmak üzere** tasarlanmıştır.
- İlk adımda Google Drive bağlanır, bu nedenle Drive erişim izni gereklidir.
- Not defterindeki proje yolları `/content/TAPETUM` dizinine göre yazılmıştır.
- Değerlendirme aşaması, birleşik bir süreçte birden fazla modelin çıktılarını karşılaştırır.

### Minimum README talimatı

README dosyasında kısa bir versiyon istiyorsanız şunu kullanabilirsiniz:

```markdown
Projeyi Google Colab'da `TAPETUM.ipynb` dosyasını kullanarak çalıştırın.

Önerilen sıra:
1. TAPETUM dosyasını Drive'dan `/content/TAPETUM` konumuna kopyalayın.
2. Tüm TAPETUM modellerini çalıştırın.
3. Tüm modelleri eğitin
4. Tüm modelleri test edin.
5. Ölçütleri değerlendirin
6. Senkronizasyon sonuçlarını Drive'a geri yükleyin.
```


## Eğitim ve Değerlendirme

Depoyu ve iş akışını aşağıdaki adımlar doğrultusunda düzenleyebilirsiniz:

```bash
python train.py
python test.py
python evaluate.py
```

Depoda bulunan değerlendirme kaynakları:

- Karşılaştırma kayıtları: `https://github.com/muratdelen/TAPETUM/tree/main/comparison_results`
- Sonuç görselleri: `https://github.com/muratdelen/TAPETUM/tree/main/LoLv2`
- Metrikler: `https://github.com/muratdelen/TAPETUM/tree/main/Metrics`

---

## İndirmeler

### GitHub Deposu
- `https://github.com/muratdelen/TAPETUM.git`

### Google Drive
- **TAPETUM İNDİRME**
`https://drive.google.com/drive/folders/1EtT9abcdGIWMrzZ2zUGHB0A_gg7LMM8J?usp=sharing`
- **VERİ KÜMESİ İNDİRME**
`https://drive.google.com/drive/folders/1QO2_buG32OjDI2w3Cg1_8e5MquEww6Ix?usp=sharing`
- **RETINEXNET İNDİRME**
`https://drive.google.com/drive/folders/1CKqjhcsQ5Fs8Btkn4jFoFXqCy9gZlh35?usp=sharing`
- **LOLV2 İNDİRME SONUCU**
`https://drive.google.com/drive/folders/1dTq0xWTz0xJL2ngVaFqajoVVtfNE2VgY?usp=sharing`
- **ÖLÇÜMLERİ İNDİR**
`https://drive.google.com/drive/folders/13XOBg-1gWTgSrbhDkDteI1pIqVIdjCfE?usp=sharing`

---

## Alıntı

Araştırmanızda bu depoyu kullanıyorsanız, şu şekilde kaynak gösterin:

```bibtex
@article{delen2026tapetum,
Başlık={Tapetum-Retinex: Düşük Işıkta Görüntü İyileştirme için Biyolojik Esintili Retinex Çerçevesi},
yazar={Delen, Murat},
yıl={2026}
}
```

---

## Yazar

**Murat Delen**
Bilgisayar Mühendisliği
Harran Üniversitesi
GitHub: `https://github.com/muratdelen`

---

## Lisans

Bu arşiv **araştırma ve akademik amaçlar** için sağlanmıştır.


---

## İlgili Çalışmalar (Düşük Işıkta Görüntü İyileştirme)

Düşük ışıkta görüntü iyileştirme (LLIE), Retinex tabanlı, öğrenme tabanlı ve eğri tabanlı yaklaşımlar kullanılarak geniş çapta incelenmiştir.

### Retinex Tabanlı Derin Modeller

Birçok derin öğrenme yöntemi, klasik Retinex teorisi üzerine kurulmuştur:

- **RetinexNet** – Yansıma ve aydınlatmayı CNN'ler kullanarak ayıran öncü bir derin Retinex ayrıştırma modeli.
- **KinD / KinD++** – Yapısal doğruluğu artırmak için aydınlatma ayarlama ve yansıma geri yükleme modülleri eklendi.
- **RUAS** – Düşük ışık koşullarında görüntü iyileştirme için hafif, denetimsiz bir mimari.

Bu modeller genellikle Retinex formülasyonunu takip eder:

```matematik
I(x) = R(x) \cdot L(x)
```

### Eğri Tabanlı Geliştirme

Bir diğer yöntem ailesi ise aydınlatma eğrilerini doğrudan öğrenir:

- **Zero-DCE / Zero-DCE++** – Eşleştirilmiş denetim olmadan piksel bazında ışık iyileştirme eğrilerini tahmin eder.

Bu yöntemler hesaplama açısından verimlidir ancak genellikle ciddi aydınlatma bozulmalarıyla başa çıkmakta zorlanırlar.

### Biyolojik İlhamlı Geliştirme

Son araştırmalar, görsel algı için biyolojik olarak esinlenmiş modelleri incelemeye başlamıştır. TAPETUM çerçevesi, **tapetum lucidum foton yansıma mekanizmasını** modelleyerek ve biyolojik ışık yükseltmesini hesaplamalı bir aydınlatma geliştirme modülüne dönüştürerek bu yöne katkıda bulunmaktadır.

Klasik Retinex bazlı yöntemlerle karşılaştırıldığında, TAPETUM şunları sunar:

- Foton yansımasından esinlenilmiş aydınlatma yükseltmesi
- Spektral kanal duyarlı geliştirme (RGB Tapetum)
- Öğrenilmiş ayrıştırma (DecomNet) ile uyumluluk

Bu, TAPETUM'un **biyolojik ilhamı derinlemesine Retinex modellemesiyle** birleştirmesine olanak tanır.

---


---

## Kıyaslama Sonuçları (`summary_metrics.csv` dosyasından düzeltilmiştir)

Aşağıdaki tablo, dışa aktarılan ölçüm özet dosyasından doğrudan güncellenmiştir.

| Model | Eşleşen Dosyalar | PSNR ↑ | SSIM ↑ | MAE ↓ | MSE ↓ | RMSE ↓ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DecomNetRetinexTapetumRGB | 100 | 19.2938 | 0.7632 | 24.6575 | 1009.2340 | 29.8147 | 0.3983 |
| DecomNetRetinexTapetum | 100 | 19.2473 | 0.7734 | 24.7627 | 997.9153 | 29.7785 | 0.3669 |
| RetinexNet | 100 | 15.9504 | 0.6524 | 0.1396 | 0.0284 | 0.1639 | Yok |
| RetinexTapetumRGB | 100 | 12.4179 | 0.4208 | 62.0526 | 4733.0982 | 65.0186 | 0.3411 |
| RetinexTapetum | 100 | 11.9131 | 0.3942 | 64.8876 | 5118.1268 | 68.1592 | 0.3541 |

### Tercüme

- **DecomNetRetinexTapetumRGB**, **19.2938** ile en yüksek ortalama **PSNR** değerine ulaşmıştır.
- **DecomNetRetinexTapetum**, **0.7734** ile en yüksek ortalama **SSIM** değerine ulaşmıştır.
- TAPETUM ailesi içinde, genel olarak en güçlü iki varyant **DecomNet tabanlı modellerdir**.
- **DecomNetRetinexTapetumRGB** ve **DecomNetRetinexTapetum** arasındaki metrik fark, daha güçlü parlaklık geri kazanımı ile daha güçlü yapısal koruma arasında bir denge olarak yorumlanabilir.
- **RetinexNet**, diğer modellere kıyasla MAE, MSE ve RMSE için farklı bir çıktı/hata ölçeği kullanır, bu nedenle bu değerler dikkatle yorumlanmalıdır.


---

# Retinex'in Biyolojik Arka Planı

İnsan Görme Yeteneği ve Retinex Teorisi

Retinex teorisi, insan görsel sisteminin değişen aydınlatma koşulları altında renkleri nasıl algıladığını açıklamak için **Edwin H. Land ve John J. McCann (1971)** tarafından ortaya atılmıştır. Basit piksel tabanlı parlaklık algısının aksine, insan gözü algılanan renk ve parlaklığı belirlemek için **tüm sahne boyunca mekansal karşılaştırmalar** yapar.

İnsan gözü, ışığı üç bağımsız koni kanalı kullanarak işler:

L, M, S

Uzun, orta ve kısa dalga boylu tepkileri temsil eder.

Retinex modelleri, bir görüntüyü yansıma ve aydınlatma bileşenlerine ayırarak bu mekanizmayı taklit eder:

I(x) = R(x) L(x)

Neresi

I(x): gözlemlenen görüntü
R(x): yansıma (nesnenin doğal rengi)
L(x): aydınlatma (aydınlatma koşulları)

Bu ayrıştırma, aydınlatma normalizasyonunu ve görünürlüğün iyileştirilmesini sağlar.

## Mekansal Karşılaştırma Mekanizması

Erken dönem nörofizyolojik çalışmalar, retina nöronlarının görsel uyaranları işlerken **merkez-çevre mekansal karşılaştırmaları** yaptığını göstermiştir.

Önemli katkılar şunlardır:

- **Kuffler (1953)** – merkez-çevre alıcı alanlarının keşfi
- **Barlow** – algıda mekansal karşılaştırma mekanizmaları
- **Hubel ve Wiesel** – görsel kortekste mekansal özellik tespiti

Bu çalışmalar, algının mutlak parlaklığa değil, **göreceli mekansal farklılıklara** bağlı olduğunu göstermiştir.

Retinex algoritmaları, yerel kontrast hesaplamaları yoluyla bu süreci taklit eder.

## Renk Mondrian Deneyi

Land'in ünlü **Renk Mondrian deneyi**, algılanan rengin mutlak spektral ölçümlerden ziyade mekansal ilişkilere daha çok bağlı olduğunu göstermiştir.

Aynı fiziksel ışık yoğunluğunu yansıtan iki yüzey, çevrelerindeki bölgelere bağlı olarak **farklı renklerde** görünebilir.

Bu, renk algısının mutlak ışık değerlerinden ziyade **göreceli mekansal karşılaştırmalar** tarafından belirlendiğini doğrulamaktadır.

## Düşük Işıkta Görüntü İyileştirme ile İlişkisi

Düşük ışıkta görüntü iyileştirme, içsel yansımayı korurken aydınlatma bileşenini geri kazanmak olarak yorumlanabilir.

Retinex bazlı modellerin tahmini:

- sahne aydınlatması
- içsel yansıma

ve daha iyi görünürlüğe sahip, geliştirilmiş bir görüntü yeniden oluşturulur.

## TAPETUM Çerçevesine Bağlantı

**TAPETUM çerçevesi**, gece hayvanlarının gözlerinde bulunan yansıtıcı bir tabaka olan **tapetum lucidum**'dan esinlenerek geliştirilen biyolojik bir aydınlatma yükseltme mekanizmasını bünyesine katarak klasik Retinex'i genişletiyor.

Tapetum lucidum, gelen ışığı retinadan geri yansıtarak düşük ışık koşullarında foton yakalama oranını artırır.

TAPETUM aydınlatma modeli, aydınlatmayı şu şekilde değiştirir:

Lt(x) = L(x)(1 + λT(x))

Neresi

T(x): Tapetum dikkat haritası
λ: yükseltme gücü

## RGB Spektral Uyarlaması (Ren Geyiği Görüşünden İlham Alınarak)

Ren geyiği gibi bazı hayvanlarda tapetum lucidum'un yansıtıcı özelliklerinde mevsimsel değişiklikler görülür.

Kış aylarında tapetum daha fazla **mavi dalga boyunu** yansıtarak düşük ışıklı ortamlarda görme hassasiyetini artırır.

Bu olgudan ilham alan **Retinex ‑ Tapetum RGB**, kanal bazında aydınlatma yükseltmesi gerçekleştirir:

Lc(x) = Lc(x)(1 + λTc(x))

∈ {R,G,B} için .

Bu spektral amplifikasyon parlaklığı artırabilir (daha yüksek PSNR), ancak yapısal benzerliği biraz bozabilir (daha düşük SSIM), bu da deneylerde gözlemlenen metrik farklılıkları açıklar.
