@TestOn('vm')
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_scene/scene.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/fmat/fmat.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math.dart';

import 'support/gpu_available.dart';

// The source with `//` comments removed, so prose mentioning discard does not
// count.
String _code(String source) =>
    source.split('\n').map((line) => line.split('//').first).join('\n');

const _size = ui.Size(100, 100);

// Draws one mesh of two overlapping quads in [alphaMode]: the far quad comes
// first in the index order, so in the coverage color draw its fragments fail
// the equal depth test before the near quad's fragments shade.
Future<List<int>> _overlappingQuadsCentre(AlphaMode alphaMode) => _quadsCentre(
  PhysicallyBasedMaterial()
    ..baseColorFactor = Vector4(0.1, 0.8, 0.2, 1)
    ..alphaMode = alphaMode
    ..doubleSided = true,
);

// [_overlappingQuadsCentre] with any [material]; the centre pixel's RGBA.
Future<List<int>> _quadsCentre(Material material) async {
  await Scene.initializeStaticResources();
  final scene = Scene();
  final positions = <double>[
    for (final z in const [1.0, -1.0]) ...[
      -1, -1, z, 1, -1, z, 1, 1, z, -1, 1, z, //
    ],
  ];
  scene.add(
    Node(
      mesh: Mesh(
        MeshGeometry.fromArrays(
          positions: Float32List.fromList(positions),
          indices: const [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
        ),
        material,
      ),
    ),
  );
  final recorder = ui.PictureRecorder();
  scene.render(
    PerspectiveCamera(),
    ui.Canvas(recorder),
    viewport: ui.Offset.zero & _size,
    pixelRatio: 1.0,
  );
  final image = await recorder.endRecording().toImage(
    _size.width.toInt(),
    _size.height.toInt(),
  );
  final bytes = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!;
  image.dispose();
  final offset =
      ((_size.height ~/ 2) * _size.width.toInt() + _size.width ~/ 2) * 4;
  return [
    for (var channel = 0; channel < 4; channel++)
      bytes.getUint8(offset + channel),
  ];
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final noGpu = gpuAvailable()
      ? null
      : 'Requires a GPU device: --enable-impeller --enable-flutter-gpu.';

  test('the opaque material shaders never discard', () {
    // A discard anywhere in a shader turns off early depth testing and
    // hidden-surface removal for its every draw on tiled GPUs; cutouts go
    // through the coverage pre-draw instead.
    for (final path in [
      'shaders/flutter_scene_standard.frag',
      'shaders/flutter_scene_unlit.frag',
    ]) {
      final code = _code(File(path).readAsStringSync());
      expect(code, isNot(contains('discard')), reason: path);
      expect(code, isNot(contains('ApplyLodFade')), reason: path);
    }
    const physical = 'assets/materials/physical_opaque.fmat';
    final compiled = compileFmat(
      File(physical).readAsStringSync(),
      fileName: physical,
    );
    expect(_code(compiled.glsl), isNot(contains('discard')));
  });

  test('the coverage pre-draw applies the cross-fade and the mask', () {
    final code = _code(
      File('shaders/flutter_scene_coverage.frag').readAsStringSync(),
    );
    expect(code, contains('ApplyLodFade(coverage_info.fade)'));
    expect(code, contains('ApplyDepthAlphaMask()'));
    final bundle =
        jsonDecode(File('shaders/base.shaderbundle.json').readAsStringSync())
            as Map<String, Object?>;
    expect(bundle['CoverageFragment'], {
      'type': 'fragment',
      'file': 'shaders/flutter_scene_coverage.frag',
    });
  });

  test('the built-in lit and unlit materials cross-fade', () {
    expect(PhysicallyBasedMaterial().lodCrossFades, isTrue);
    expect(UnlitMaterial().lodCrossFades, isTrue);
  });

  test('an alpha-masked material cuts out through its depth mask', () {
    final material = PhysicallyBasedMaterial();
    expect(material.depthAlphaMasked, isFalse);
    material.alphaMode = AlphaMode.mask;
    expect(material.depthAlphaMasked, isTrue);
  });

  test('an alpha-masked unlit material is opaque and cuts out through its '
      'depth mask', () {
    final material = UnlitMaterial();
    expect(material.depthAlphaMasked, isFalse);
    expect(material.alphaCutoff, 0.5);
    material.alphaMode = AlphaMode.mask;
    expect(material.depthAlphaMasked, isTrue);
    expect(material.isOpaque(), isTrue);
    material.alphaMode = AlphaMode.blend;
    expect(material.depthAlphaMasked, isFalse);
    expect(material.isOpaque(), isFalse);
  });

  test('an alpha-masked mesh whose far triangles come first in its index '
      'order draws its near triangles where they are kept, as the opaque '
      'mesh does', () async {
    final opaque = await _overlappingQuadsCentre(AlphaMode.opaque);
    final masked = await _overlappingQuadsCentre(AlphaMode.mask);
    expect(opaque[3], 255);
    for (var channel = 0; channel < 4; channel++) {
      expect(
        (masked[channel] - opaque[channel]).abs(),
        lessThanOrEqualTo(2),
        reason: 'masked $masked against opaque $opaque',
      );
    }
  }, skip: noGpu);
  test('an alpha-masked unlit material keeps what reaches its cutoff at full '
      'opacity and cuts away the rest', () async {
    UnlitMaterial unlit(AlphaMode mode, double alpha) => UnlitMaterial()
      ..baseColorFactor = Vector4(0.1, 0.8, 0.2, alpha)
      ..vertexColorWeight = 0
      ..alphaMode = mode
      ..doubleSided = true;
    final opaque = await _quadsCentre(unlit(AlphaMode.opaque, 1));
    final kept = await _quadsCentre(unlit(AlphaMode.mask, 0.7));
    final cut = await _quadsCentre(unlit(AlphaMode.mask, 0.3));
    final lowCutoff = await _quadsCentre(
      unlit(AlphaMode.mask, 0.3)..alphaCutoff = 0.2,
    );
    expect(opaque[3], 255);
    for (var channel = 0; channel < 4; channel++) {
      expect(
        (kept[channel] - opaque[channel]).abs(),
        lessThanOrEqualTo(2),
        reason: 'kept $kept against opaque $opaque',
      );
      expect(
        (lowCutoff[channel] - opaque[channel]).abs(),
        lessThanOrEqualTo(2),
        reason: 'cutoff 0.2 $lowCutoff against opaque $opaque',
      );
    }
    expect(cut[3], 0, reason: 'cut $cut');
  }, skip: noGpu);
}
