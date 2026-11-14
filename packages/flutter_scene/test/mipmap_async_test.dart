// Covers the off-thread mip chain build: the chain built on a background
// isolate is byte for byte the chain the synchronous builder produces, for
// every content kind and a non-square size, and `Texture2D.fromImage` (which
// builds its chain that way) still uploads a full chain. The upload half is
// GPU-gated like the other render suites.

import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_scene/scene.dart';
import 'package:flutter_scene/src/texture/mipmap.dart';
import 'package:flutter_scene/src/texture/mipmap_async.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/gpu_available.dart';

// A deterministic non-trivial RGBA image: every channel varies with position so
// a swapped channel, row or level would show.
Uint8List _pattern(int w, int h) {
  final p = Uint8List(w * h * 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final i = (y * w + x) * 4;
      p[i] = (x * 7 + y * 3) & 0xFF;
      p[i + 1] = (x * 5 + y * 11 + 40) & 0xFF;
      p[i + 2] = (x * 13 + y * 2 + 90) & 0xFF;
      p[i + 3] = (x + y) & 1 == 0 ? 255 : 128;
    }
  }
  return p;
}

Future<ui.Image> _imageFromPixels(Uint8List pixels, int w, int h) {
  final completer = Completer<ui.Image>();
  ui.decodeImageFromPixels(
    pixels,
    w,
    h,
    ui.PixelFormat.rgba8888,
    completer.complete,
  );
  return completer.future;
}

void main() {
  for (final content in TextureContent.values) {
    test(
      'async chain equals the sync chain for ${content.name} content',
      () async {
        const w = 48;
        const h = 20; // non-square, not a power of two
        final pixels = _pattern(w, h);
        final sync = generateMipChain(pixels, w, h, content);
        final async = await generateMipChainAsync(pixels, w, h, content);

        expect(async.length, sync.length);
        for (var i = 0; i < sync.length; i++) {
          expect(async[i].width, sync[i].width, reason: 'level $i width');
          expect(async[i].height, sync[i].height, reason: 'level $i height');
          expect(async[i].pixels, equals(sync[i].pixels), reason: 'level $i');
        }
      },
    );
  }

  test('the async build leaves the caller\'s pixels untouched', () async {
    final pixels = _pattern(16, 16);
    final copy = Uint8List.fromList(pixels);
    await generateMipChainAsync(pixels, 16, 16, TextureContent.color);
    expect(pixels, equals(copy));
  });

  if (!gpuAvailable()) {
    test(
      'fromImage uploads a full mip chain (skipped: no GPU device)',
      () {},
      skip: 'Requires a GPU device.',
    );
    return;
  }

  testWidgets('fromImage uploads the same chain length as fromPixels', (
    tester,
  ) async {
    const w = 32;
    const h = 16;
    final pixels = _pattern(w, h);
    await tester.runAsync(() async {
      final image = await _imageFromPixels(pixels, w, h);
      final fromImage = await Texture2D.fromImage(image);
      image.dispose();
      final fromPixels = Texture2D.fromPixels(pixels, w, h);
      expect(
        fromImage.gpuTexture.mipLevelCount,
        fromPixels.gpuTexture.mipLevelCount,
      );
      expect(fromImage.gpuTexture.mipLevelCount, greaterThan(1));
    });
  });
}
