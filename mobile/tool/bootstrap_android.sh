#!/usr/bin/env bash
set -euo pipefail

# Generate Flutter's Android host and apply the tracked ONNX Runtime bridge.
flutter create --platforms=android --project-name retinextapetum_camera --org io.github.muratdelen .
python3 tool/configure_android.py
