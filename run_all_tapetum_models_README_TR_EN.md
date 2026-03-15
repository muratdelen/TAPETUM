# TAPETUM Models Comparison Pipeline

------------------------------------------------------------------------

## Turkish

# Run All TAPETUM Models -- Colab Kullanımı

Bu script 4 TAPETUM modelini aynı deney standardında sırayla çalıştırmak
için hazırlanmıştır.

Modeller: - RetinexTapetum - RetinexTapetumRGB -
DecomNetRetinexTapetum - DecomNetRetinexTapetumRGB

Amaç: tüm modelleri **aynı koşullarda eğitip test ederek** sonuçlarını
adil şekilde karşılaştırmaktır.

## Beklenen klasör yapısı

/content/TAPETUM/ RetinexTapetum/ RetinexTapetumRGB/
DecomNetRetinexTapetum/ DecomNetRetinexTapetumRGB/

Her model klasöründe en az şu dosyalar bulunmalıdır:

config.py train.py test.py

## Script ne yapar?

Script şu işlemleri otomatik yapar:

1.  Her modelin config.py içindeki EPOCHS değerini günceller.
2.  train.py çalıştırır.
3.  history.csv varsa en iyi epoch ve en iyi validation PSNR değerini
    okur.
4.  test.py çalıştırır.
5.  Oluşan metrik CSV dosyalarını arar.
6.  Tüm sonuçları tek bir CSV ve JSON dosyasında toplar.

## Colab kullanımı

1)  Google Drive bağla

from google.colab import drive drive.mount('/content/drive')

2)  Proje klasörünü kopyala

!cp -r /content/drive/MyDrive/TAPETUM /content/TAPETUM

3)  Scripti yükle

run_all_tapetum_models_colab.py dosyasını Colab'a yükle.

4)  Scripti çalıştır

!python /content/run_all_tapetum_models_colab.py

## Değiştirilebilecek parametreler

Script başında şu değişkenler bulunur:

ROOT TARGET_EPOCHS RUN_TRAIN RUN_TEST STOP_ON_ERROR

## Üretilen çıktılar

/content/TAPETUM/comparison_results/

all_models_summary.csv all_models_summary.json

Her model için ayrıca: train_log.txt test_log.txt history.csv

## Not

Bazı test scriptleri farklı isimde CSV üretebilir. Script mümkün
olduğunca esnek arama yapar fakat en sağlıklı yöntem tüm modellerde test
CSV adlarını standartlaştırmaktır.

------------------------------------------------------------------------

## English

# Run All TAPETUM Models -- Colab Usage

This script runs four TAPETUM models sequentially under the same
experimental setup.

Models: - RetinexTapetum - RetinexTapetumRGB - DecomNetRetinexTapetum -
DecomNetRetinexTapetumRGB

The goal is to train and test all models under the **same conditions**
to allow fair comparison.

## Expected folder structure

/content/TAPETUM/ RetinexTapetum/ RetinexTapetumRGB/
DecomNetRetinexTapetum/ DecomNetRetinexTapetumRGB/

Each model directory should contain at least:

config.py train.py test.py

## What the script does

The script automatically performs the following steps:

1.  Updates the EPOCHS value inside each model's config.py.
2.  Runs training using train.py.
3.  Reads history.csv (if available) to extract:
    -   best epoch
    -   best validation PSNR
4.  Runs testing using test.py.
5.  Searches for generated metric CSV files.
6.  Collects all results into a single CSV and JSON file.

## Colab usage

1)  Mount Google Drive

from google.colab import drive drive.mount('/content/drive')

2)  Copy the project folder

!cp -r /content/drive/MyDrive/TAPETUM /content/TAPETUM

3)  Upload the script

Upload run_all_tapetum_models_colab.py to Colab.

4)  Run the script

!python /content/run_all_tapetum_models_colab.py

## Parameters you can modify

At the beginning of the script:

ROOT TARGET_EPOCHS RUN_TRAIN RUN_TEST STOP_ON_ERROR

## Generated outputs

/content/TAPETUM/comparison_results/

all_models_summary.csv all_models_summary.json

For each model: train_log.txt test_log.txt history.csv

## Note

Some test scripts may generate CSV files with different names. The
script tries to detect them automatically, but the most reliable
approach is to standardize the test CSV filenames across all models.
