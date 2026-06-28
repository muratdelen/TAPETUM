import 'dart:typed_data';

import 'package:flutter/material.dart';

class BeforeAfterSlider extends StatefulWidget {
  const BeforeAfterSlider({
    super.key,
    required this.before,
    required this.after,
  });

  final Uint8List before;
  final Uint8List after;

  @override
  State<BeforeAfterSlider> createState() => _BeforeAfterSliderState();
}

class _BeforeAfterSliderState extends State<BeforeAfterSlider> {
  double _position = 0.5;

  void _updatePosition(Offset localPosition, double width) {
    setState(() {
      _position = (localPosition.dx / width).clamp(0.0, 1.0);
    });
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        return AspectRatio(
          aspectRatio: 3 / 4,
          child: GestureDetector(
            onHorizontalDragUpdate: (details) => _updatePosition(details.localPosition, constraints.maxWidth),
            onTapDown: (details) => _updatePosition(details.localPosition, constraints.maxWidth),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(20),
              child: Stack(
                fit: StackFit.expand,
                children: <Widget>[
                  Image.memory(widget.before, fit: BoxFit.cover),
                  ClipRect(
                    clipper: _SplitClipper(_position),
                    child: Image.memory(widget.after, fit: BoxFit.cover),
                  ),
                  Align(
                    alignment: Alignment(_position * 2 - 1, 0),
                    child: Container(width: 3, color: Colors.white),
                  ),
                  Align(
                    alignment: Alignment(_position * 2 - 1, 0),
                    child: Container(
                      width: 42,
                      height: 42,
                      decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle),
                      child: const Icon(Icons.compare_arrows, color: Colors.black87),
                    ),
                  ),
                  const Positioned(top: 12, left: 12, child: _Badge(text: 'Original')),
                  const Positioned(top: 12, right: 12, child: _Badge(text: 'Enhanced')),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class _Badge extends StatelessWidget {
  const _Badge({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(color: Colors.black54, borderRadius: BorderRadius.circular(99)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 12)),
      ),
    );
  }
}

class _SplitClipper extends CustomClipper<Rect> {
  const _SplitClipper(this.fraction);

  final double fraction;

  @override
  Rect getClip(Size size) => Rect.fromLTWH(0, 0, size.width * fraction, size.height);

  @override
  bool shouldReclip(covariant _SplitClipper oldClipper) => oldClipper.fraction != fraction;
}
