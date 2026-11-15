/// Covers the table-driven linear-to-sRGB byte conversion of the color mip
/// chain: it must return exactly the byte of the reference formula (the `pow`
/// the chain used before), so a chain built with it is byte-for-byte the
/// chain built before.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_scene/src/texture/mipmap.dart';
import 'package:test/test.dart';

double _srgbToLinear(int byte) {
  final c = byte / 255.0;
  return c <= 0.04045
      ? c / 12.92
      : math.pow((c + 0.055) / 1.055, 2.4).toDouble();
}

final _bits = ByteData(8);
double _fromBits(int b) => (_bits..setInt64(0, b)).getFloat64(0);
int _toBits(double d) => (_bits..setFloat64(0, d)).getInt64(0);

/// The color chain as built before the table: the same 2x2 box in linear
/// light, rounded back through [linearToSrgbByteReference].
List<Uint8List> _referenceChain(Uint8List pixels, int width, int height) {
  final table = [for (var b = 0; b < 256; b++) _srgbToLinear(b)];
  final out = <Uint8List>[pixels];
  var w = width, h = height;
  var src = pixels;
  while (w > 1 || h > 1) {
    final nw = math.max(1, w >> 1), nh = math.max(1, h >> 1);
    final dst = Uint8List(nw * nh * 4);
    for (var y = 0; y < nh; y++) {
      final y0 = y * 2 < h ? y * 2 : h - 1;
      final y1 = y0 + 1 < h ? y0 + 1 : h - 1;
      for (var x = 0; x < nw; x++) {
        final x0 = x * 2 < w ? x * 2 : w - 1;
        final x1 = x0 + 1 < w ? x0 + 1 : w - 1;
        final a = (y0 * w + x0) * 4, b = (y0 * w + x1) * 4;
        final c = (y1 * w + x0) * 4, d = (y1 * w + x1) * 4;
        final o = (y * nw + x) * 4;
        for (var ch = 0; ch < 3; ch++) {
          final avg =
              (table[src[a + ch]] +
                  table[src[b + ch]] +
                  table[src[c + ch]] +
                  table[src[d + ch]]) *
              0.25;
          dst[o + ch] = linearToSrgbByteReference(avg);
        }
        dst[o + 3] =
            ((src[a + 3] + src[b + 3] + src[c + 3] + src[d + 3]) + 2) ~/ 4;
      }
    }
    out.add(dst);
    src = dst;
    w = nw;
    h = nh;
  }
  return out;
}

void main() {
  test('table byte equals the reference around every step', () {
    // Each byte step k of the reference, found by bisection, then every
    // double within 4096 ulps of it.
    var lo0 = 0;
    var checked = 0;
    for (var k = 1; k < 256; k++) {
      var lo = lo0, hi = _toBits(2.0);
      while (hi - lo > 1) {
        final mid = lo + ((hi - lo) >> 1);
        if (linearToSrgbByteReference(_fromBits(mid)) >= k) {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      lo0 = lo;
      for (var b = math.max(0, hi - 4096); b <= hi + 4096; b++) {
        final x = _fromBits(b);
        expect(
          linearToSrgbByte(x),
          linearToSrgbByteReference(x),
          reason: 'x = $x (step $k)',
        );
        checked++;
      }
    }
    expect(checked, greaterThan(255 * 8000));
  });

  test('table byte equals the reference at slice edges and the ends', () {
    for (var i = 0; i <= 4096; i++) {
      final c = _toBits(i / 4096);
      for (var b = math.max(0, c - 64); b <= c + 64; b++) {
        final x = _fromBits(b);
        expect(linearToSrgbByte(x), linearToSrgbByteReference(x));
      }
    }
    for (final x in [0.0, 1e-300, 0.0031308, 1.0, 1.5, 4.0]) {
      expect(linearToSrgbByte(x), linearToSrgbByteReference(x), reason: '$x');
    }
  });

  test('table byte equals the reference on random 2x2 averages', () {
    final table = [for (var b = 0; b < 256; b++) _srgbToLinear(b)];
    final r = math.Random(43);
    for (var i = 0; i < 2000000; i++) {
      final x =
          (table[r.nextInt(256)] +
              table[r.nextInt(256)] +
              table[r.nextInt(256)] +
              table[r.nextInt(256)]) *
          0.25;
      if (linearToSrgbByte(x) != linearToSrgbByteReference(x)) {
        fail('x = $x');
      }
    }
  });

  test('color chain is byte-for-byte the pre-table chain', () {
    final r = math.Random(7);
    for (final (w, h) in [(256, 256), (300, 77), (1, 9), (129, 128)]) {
      final px = Uint8List(w * h * 4);
      for (var i = 0; i < px.length; i++) {
        px[i] = r.nextInt(256);
      }
      final got = generateMipChain(px, w, h, TextureContent.color);
      final want = _referenceChain(px, w, h);
      expect(got.length, want.length);
      for (var l = 0; l < got.length; l++) {
        expect(got[l].pixels, want[l], reason: '${w}x$h level $l');
      }
    }
  });
}
