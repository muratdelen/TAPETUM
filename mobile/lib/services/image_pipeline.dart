import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

const int modelSize = 512;

/// Prepares a captured image for the model input shape [1, 3, 512, 512].
Future<Map<String, Object>> prepareForModel(Uint8List encodedImage) {
  return compute(_prepareForModel, encodedImage);
}

Map<String, Object> _prepareForModel(Uint8List encodedImage) {
  final decoded = img.decodeImage(encodedImage);
  if (decoded == null) {
    throw StateError('The captured photo could not be decoded.');
  }

  final source = img.bakeOrientation(decoded);
  final resized = img.copyResize(
    source,
    width: modelSize,
    height: modelSize,
    interpolation: img.Interpolation.cubic,
  );

  final input = Float32List(3 * modelSize * modelSize);
  var lumaTotal = 0.0;

  for (var y = 0; y < modelSize; y++) {
    for (var x = 0; x < modelSize; x++) {
      final pixel = resized.getPixel(x, y);
      final r = pixel.r.toDouble() / 255.0;
      final g = pixel.g.toDouble() / 255.0;
      final b = pixel.b.toDouble() / 255.0;
      final index = y * modelSize + x;

      input[index] = r;
      input[modelSize * modelSize + index] = g;
      input[2 * modelSize * modelSize + index] = b;
      lumaTotal += 0.299 * r + 0.587 * g + 0.114 * b;
    }
  }

  return <String, Object>{
    'input': input,
    'width': source.width,
    'height': source.height,
    'brightness': lumaTotal / (modelSize * modelSize),
  };
}

/// Converts the NCHW model output back to a JPEG at the capture dimensions.
Future<Uint8List> encodeModelOutput({
  required Float32List output,
  required int targetWidth,
  required int targetHeight,
}) {
  return compute(_encodeModelOutput, <String, Object>{
    'output': output,
    'width': targetWidth,
    'height': targetHeight,
  });
}

Uint8List _encodeModelOutput(Map<String, Object> payload) {
  final output = payload['output']! as Float32List;
  final targetWidth = payload['width']! as int;
  final targetHeight = payload['height']! as int;
  final expected = 3 * modelSize * modelSize;

  if (output.length != expected) {
    throw StateError('Unexpected model output length: ${output.length}; expected $expected.');
  }

  final enhanced = img.Image(width: modelSize, height: modelSize, numChannels: 3);
  for (var y = 0; y < modelSize; y++) {
    for (var x = 0; x < modelSize; x++) {
      final index = y * modelSize + x;
      final r = (output[index].clamp(0.0, 1.0) * 255.0).round();
      final g = (output[modelSize * modelSize + index].clamp(0.0, 1.0) * 255.0).round();
      final b = (output[2 * modelSize * modelSize + index].clamp(0.0, 1.0) * 255.0).round();
      enhanced.setPixelRgb(x, y, r, g, b);
    }
  }

  final restored = img.copyResize(
    enhanced,
    width: targetWidth,
    height: targetHeight,
    interpolation: img.Interpolation.cubic,
  );
  return Uint8List.fromList(img.encodeJpg(restored, quality: 96));
}
