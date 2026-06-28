# Model Asset

Place the exported model at:

```text
assets/models/retinex_tapetum_512.onnx
```

The required model accepts and returns `float32` RGB tensors with shape `[1, 3, 512, 512]` and values in `[0, 1]`.

This folder contains no model binary in the source-only checkout. Add the exported ONNX file before building an inference-enabled APK.
