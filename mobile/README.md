# RetinexTapetum Camera — Android Demo

This directory contains an Android Flutter demo for the **RetinexTapetum** low-light image enhancement model.

The application captures a photo, estimates its average brightness, and runs the model **on the device** when the image is sufficiently dark. No image is sent to a server. The result screen provides a before/after comparison and lets the user save the enhanced image to the device gallery.

> **Status:** Research prototype. Android is supported in this repository. The iOS inference bridge has not been implemented yet.

> **Model file:** The source project expects `assets/models/retinex_tapetum_512.onnx`. This binary must be added before publishing an inference-enabled test build. The repository source and Android build workflow can be inspected without it.

## Features

- Camera capture with the rear camera by default
- Automatic dark-image detection; optionally process every capture
- On-device ONNX Runtime inference
- Before/after comparison slider
- Save the enhanced image to the device gallery
- Public GitHub Actions workflows for build verification and APK releases

## Model I/O

| Item | Value |
|---|---|
| Model file | `assets/models/retinex_tapetum_512.onnx` |
| Runtime | ONNX Runtime Android |
| Input | RGB `float32`, shape `[1, 3, 512, 512]`, values in `[0, 1]` |
| Output | RGB `float32`, shape `[1, 3, 512, 512]`, values in `[0, 1]` |
| Inference device | Android device only; no server request |

The Python training and evaluation code remain in the repository root. This mobile application wraps the exported inference model and returns only the final enhanced image.

## Run Locally

### Requirements

- Flutter SDK installed and available in your terminal
- Android Studio with an Android SDK and either a physical Android device or an emulator
- A device running Android 7.0 (API 24) or newer

### Commands

```bash
cd mobile
flutter pub get
bash tool/bootstrap_android.sh
flutter run
```

`tool/bootstrap_android.sh` creates the standard Flutter Android host project and adds the small Kotlin bridge that runs ONNX Runtime. It is safe to run again after a clean checkout.

## Build an APK

```bash
cd mobile
flutter pub get
bash tool/bootstrap_android.sh
flutter build apk --release
```

The APK is written to:

```text
build/app/outputs/flutter-apk/app-release.apk
```

## Test from GitHub

The workflow `.github/workflows/mobile-android.yml` builds a debug APK after changes to the mobile project and uploads it as an Actions artifact.

For a public installable release, push a tag with the `mobile-v` prefix:

```bash
git tag mobile-v0.1.0
git push origin mobile-v0.1.0
```

The `mobile-android-release.yml` workflow builds a release APK and attaches it to the matching GitHub Release. Anyone can download that APK from the repository Releases page and install it after allowing installs from that source on Android.

## Project Layout

```text
mobile/
├── assets/models/                 # Exported ONNX inference model
├── lib/                           # Flutter UI, image preparation, and platform bridge
├── native_templates/android/      # Kotlin ONNX Runtime integration template
├── tool/                          # Android host generation/configuration scripts
├── pubspec.yaml
└── README.md
```

## Notes and Limitations

- The current implementation resizes each capture to `512 × 512` for inference, then resizes the enhanced result back to the original photo dimensions. This is fast enough for an initial research demo, but it can lose fine detail on very high-resolution photos.
- The model is invoked after capture, not continuously for every camera preview frame.
- Enhancement is skipped when average image brightness is at least `0.42`, unless the user enables processing for every photo.
- The original image is retained in memory for comparison and is never overwritten by the application.

## Citation

If you use this demo or the model in academic work, please cite the main RetinexTapetum project:

```bibtex
@article{delen2026retinextapetum,
  title={RetinexTapetum: Bio-Inspired Active Illumination Modeling for Efficient Low-Light Image Enhancement},
  author={Delen, Murat and Ciftci, Serdar},
  year={2026}
}
```