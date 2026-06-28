import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Flutter-to-Android bridge for on-device ONNX Runtime inference.
class RetinexInference {
  RetinexInference._();

  static const _channel = MethodChannel('retinex_tapetum/inference');

  static Future<Float32List> enhance(Float32List input) async {
    final result = await _channel.invokeMethod<Object>('enhance', <String, Object>{
      'input': input,
    });

    if (result is Float32List) return result;
    if (result is List<Object?>) {
      return Float32List.fromList(result.cast<num>().map((value) => value.toDouble()).toList());
    }
    throw PlatformException(
      code: 'unexpected_output',
      message: 'Model output could not be converted to Float32List.',
    );
  }
}
