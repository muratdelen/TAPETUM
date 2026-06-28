import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image_gallery_saver_plus/image_gallery_saver_plus.dart';

import 'services/image_pipeline.dart';
import 'services/retinex_inference.dart';
import 'widgets/before_after_slider.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(RetinexTapetumCameraApp(cameras: cameras));
}

class RetinexTapetumCameraApp extends StatelessWidget {
  const RetinexTapetumCameraApp({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'RetinexTapetum Camera',
      theme: ThemeData(useMaterial3: true, brightness: Brightness.dark),
      home: CameraPage(cameras: cameras),
    );
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  CameraController? _controller;
  bool _isProcessing = false;
  bool _onlyDarkPhotos = true;
  String? _error;

  CameraDescription get _backCamera => widget.cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => widget.cameras.first,
      );

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _startCamera();
  }

  Future<void> _startCamera() async {
    final previous = _controller;
    if (previous != null) await previous.dispose();

    final controller = CameraController(
      _backCamera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    try {
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _controller = controller;
        _error = null;
      });
    } on CameraException catch (error) {
      if (!mounted) return;
      setState(() => _error = 'Unable to start the camera: ${error.description ?? error.code}');
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      controller.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _startCamera();
    }
  }

  Future<void> _takePhoto() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _isProcessing) return;

    setState(() => _isProcessing = true);
    try {
      final picture = await controller.takePicture();
      final original = await picture.readAsBytes();
      final prepared = await prepareForModel(original);
      final brightness = prepared['brightness']! as double;
      final shouldEnhance = !_onlyDarkPhotos || brightness < 0.42;

      Uint8List enhanced = original;
      if (shouldEnhance) {
        final output = await RetinexInference.enhance(prepared['input']! as Float32List);
        enhanced = await encodeModelOutput(
          output: output,
          targetWidth: prepared['width']! as int,
          targetHeight: prepared['height']! as int,
        );
      }

      if (!mounted) return;
      await Navigator.of(context).push<void>(
        MaterialPageRoute<void>(
          builder: (context) => ResultPage(
            original: original,
            enhanced: enhanced,
            enhancementApplied: shouldEnhance,
            brightness: brightness,
          ),
        ),
      );
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('The photo could not be processed: $error')),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: <Widget>[
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 18, 20, 10),
              child: Row(
                children: <Widget>[
                  const Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: <Widget>[
                        Text('RetinexTapetum Camera', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700)),
                        SizedBox(height: 3),
                        Text('On-device low-light photo enhancement'),
                      ],
                    ),
                  ),
                  Switch(
                    value: _onlyDarkPhotos,
                    onChanged: (value) => setState(() => _onlyDarkPhotos = value),
                  ),
                ],
              ),
            ),
            Expanded(
              child: Center(
                child: _error != null
                    ? Padding(
                        padding: const EdgeInsets.all(24),
                        child: Text(_error!, textAlign: TextAlign.center),
                      )
                    : controller == null || !controller.value.isInitialized
                        ? const CircularProgressIndicator()
                        : ClipRRect(
                            borderRadius: BorderRadius.circular(20),
                            child: AspectRatio(
                              aspectRatio: controller.value.aspectRatio,
                              child: CameraPreview(controller),
                            ),
                          ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 14, 24, 26),
              child: Column(
                children: <Widget>[
                  Text(
                    _onlyDarkPhotos
                        ? 'Enhance only dark photos automatically'
                        : 'Enhance every photo automatically',
                  ),
                  const SizedBox(height: 14),
                  SizedBox(
                    width: 78,
                    height: 78,
                    child: FilledButton(
                      onPressed: _isProcessing ? null : _takePhoto,
                      style: FilledButton.styleFrom(shape: const CircleBorder()),
                      child: _isProcessing
                          ? const SizedBox(width: 26, height: 26, child: CircularProgressIndicator(strokeWidth: 3))
                          : const Icon(Icons.camera_alt, size: 31),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ResultPage extends StatefulWidget {
  const ResultPage({
    super.key,
    required this.original,
    required this.enhanced,
    required this.enhancementApplied,
    required this.brightness,
  });

  final Uint8List original;
  final Uint8List enhanced;
  final bool enhancementApplied;
  final double brightness;

  @override
  State<ResultPage> createState() => _ResultPageState();
}

class _ResultPageState extends State<ResultPage> {
  bool _saving = false;

  Future<void> _save() async {
    setState(() => _saving = true);
    try {
      final name = 'retinextapetum_${DateTime.now().millisecondsSinceEpoch}';
      await ImageGallerySaverPlus.saveImage(widget.enhanced, quality: 96, name: name);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Enhanced photo saved to the gallery.')));
      }
    } catch (error) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('The photo could not be saved: $error')));
      }
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final message = widget.enhancementApplied
        ? 'Low-light enhancement was applied on this device.'
        : 'The photo was bright enough, so the original was retained.';
    return Scaffold(
      appBar: AppBar(title: const Text('Result')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            children: <Widget>[
              BeforeAfterSlider(before: widget.original, after: widget.enhanced),
              const SizedBox(height: 18),
              Text(message, textAlign: TextAlign.center),
              Text('Average brightness: ${widget.brightness.toStringAsFixed(2)}'),
              const Spacer(),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: _saving ? null : _save,
                  icon: _saving
                      ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.save_alt),
                  label: const Text('Save enhanced photo'),
                ),
              ),
              const SizedBox(height: 10),
              SizedBox(
                width: double.infinity,
                child: OutlinedButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Take another photo'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
