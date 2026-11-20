// WR-88c (home3d): a render scale that drops while the camera moves renders
// into the top-left sub-rectangle of the full-resolution targets the view
// already holds, instead of allocating a second set of targets at the smaller
// size. The image matches the old separate-targets path; a frame running a
// pass that does not honor the sub-rectangle falls back to it. GPU-gated.

// ignore_for_file: invalid_use_of_internal_member, implementation_imports

import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_scene/scene.dart';
import 'package:flutter_scene/src/render/frame_transients.dart';
import 'package:flutter_scene/src/render/render_graph.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math.dart';

import 'support/gpu_available.dart';

const _w = 128, _h = 64;

Scene _scene({AntiAliasingMode aa = AntiAliasingMode.smaa}) {
  final scene = Scene()
    ..antiAliasingMode = aa
    ..add(
      Node()..addComponent(
        DirectionalLightComponent(
          DirectionalLight(castsShadow: true, shadowMapResolution: 256),
        ),
      ),
    )
    ..add(
      Node(
        mesh: Mesh(
          CuboidGeometry(Vector3.all(1)),
          PhysicallyBasedMaterial()
            ..baseColorFactor = Vector4(0.8, 0.3, 0.1, 1),
        ),
      )..rotation = Quaternion.axisAngle(Vector3(0, 1, 0), 0.6),
    )
    ..add(
      Node(
        mesh: Mesh(
          PlaneGeometry(width: 10, depth: 10),
          PhysicallyBasedMaterial()
            ..baseColorFactor = Vector4(0.2, 0.6, 0.3, 1),
        ),
      )..position = Vector3(0, -0.5, 0),
    );
  return scene;
}

/// Renders [scene] at [scale] into a [_w] x [_h] region, waits for the GPU
/// work, and returns the composited canvas pixels.
Future<Uint8List> _frame(
  Scene scene,
  double scale, {
  double width = _w * 1.0,
}) async {
  scene.renderScale = scale;
  final recorder = ui.PictureRecorder();
  scene.renderViews(
    [
      RenderView(
        camera: PerspectiveCamera(
          position: Vector3(1.5, 2.5, 4),
          target: Vector3.zero(),
        ),
      ),
    ],
    ui.Canvas(recorder),
    region: ui.Rect.fromLTWH(0, 0, width, _h * 1.0),
    pixelRatio: 1.0,
  );
  final image = await recorder.endRecording().toImage(_w, _h);
  final bytes = (await image.toByteData(format: ui.ImageByteFormat.rawRgba))!;
  image.dispose();
  for (
    var i = 0;
    i < 100 &&
        rendererSubmissions.completedThrough <
            rendererSubmissions.latestSubmission;
    i++
  ) {
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
  return bytes.buffer.asUint8List();
}

/// Renders a few frames so lazily loaded resources (SMAA lookup tables)
/// settle, returning the last frame's pixels.
Future<Uint8List> _settled(Scene scene, double scale) async {
  late Uint8List bytes;
  for (var i = 0; i < 4; i++) {
    bytes = await _frame(scene, scale);
  }
  return bytes;
}

/// The widths of the view's size-dependent pooled targets.
Set<int> _pooledWidths(Scene scene) => {
  for (final d in scene.surface.transientTexturePool().debugHeld.keys)
    if (d.width >= _w ~/ 4) d.width,
};

/// Pixels differing by more than [tolerance] in any channel.
int _differing(Uint8List a, Uint8List b, {int tolerance = 0}) {
  var count = 0;
  for (var p = 0; p < a.length; p += 4) {
    for (var c = 0; c < 4; c++) {
      if ((a[p + c] - b[p + c]).abs() > tolerance) {
        count++;
        break;
      }
    }
  }
  return count;
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  if (!gpuAvailable()) {
    test(
      'render extent suite (skipped: no GPU device)',
      () {},
      skip: 'Requires a GPU device.',
    );
    return;
  }

  setUpAll(Scene.initializeStaticResources);

  test('T-WR88c-1 · rest -> motion -> rest keeps one set of targets and '
      'allocates nothing in motion', () async {
    final scene = _scene();
    await _settled(scene, 1.0);
    final held = scene.surface.debugHeldTextureCount;
    final created = TransientTexturePool.texturesCreated;
    await _settled(scene, 0.5);
    expect(scene.surface.lastSwapchainExtent(), const ui.Size(_w / 2, _h / 2));
    expect(_pooledWidths(scene), {_w});
    expect(TransientTexturePool.texturesCreated, created);
    await _settled(scene, 1.0);
    expect(
      scene.surface.lastSwapchainExtent(),
      const ui.Size(_w * 1.0, _h * 1.0),
    );
    expect(_pooledWidths(scene), {_w});
    expect(scene.surface.debugHeldTextureCount, held);
    expect(TransientTexturePool.texturesCreated, created);
    scene.dispose();
  });

  test('T-WR88c-2 · the sub-rectangle frame matches the separate-targets '
      'frame (SMAA, vignette)', () async {
    final viaSubrect = _scene();
    viaSubrect.postProcess.vignette.enabled = true;
    await _settled(viaSubrect, 1.0);
    final subrect = await _settled(viaSubrect, 0.5);
    expect(_pooledWidths(viaSubrect), {_w});

    final separate = _scene();
    separate.postProcess.vignette.enabled = true;
    final reference = await _settled(separate, 0.5);
    expect(_pooledWidths(separate), {_w ~/ 2});

    // SMAA's bilinear edge and weight fetches land on the same texel
    // positions in a target twice as wide; the filter's rounding moves a few
    // pixels by one 8-bit step (measured: 30 of 8192, at most 1).
    expect(_differing(subrect, reference, tolerance: 1), 0);
    expect(_differing(subrect, reference), lessThan(64));
    viaSubrect.dispose();
    separate.dispose();
  });

  test('T-WR88c-3 · without anti-aliasing the resolve writes the output '
      'directly and matches exactly (vignette)', () async {
    final viaSubrect = _scene(aa: AntiAliasingMode.none);
    viaSubrect.postProcess.vignette.enabled = true;
    await _settled(viaSubrect, 1.0);
    final subrect = await _settled(viaSubrect, 0.5);
    final separate = _scene(aa: AntiAliasingMode.none);
    separate.postProcess.vignette.enabled = true;
    final reference = await _settled(separate, 0.5);
    expect(_differing(subrect, reference), 0);
    viaSubrect.dispose();
    separate.dispose();
  });

  test('T-WR88c-4 · a pass that fills its target (bloom) falls back to '
      'targets at the frame size', () async {
    final scene = _scene()..postProcess.bloom.enabled = true;
    await _settled(scene, 1.0);
    await _settled(scene, 0.5);
    expect(_pooledWidths(scene), containsAll(<int>{_w ~/ 2}));
    expect(scene.surface.lastSwapchainExtent(), const ui.Size(_w / 2, _h / 2));
    scene.dispose();
  });

  test('T-WR88c-5 · a larger size releases the smaller one it now hosts '
      '(motion first, then rest)', () async {
    final restOnly = _scene();
    await _settled(restOnly, 1.0);
    final scene = _scene();
    await _settled(scene, 0.5);
    expect(_pooledWidths(scene), {_w ~/ 2});
    await _settled(scene, 1.0);
    expect(
      scene.surface.debugHeldTextureCount,
      restOnly.surface.debugHeldTextureCount,
    );
    await _settled(scene, 0.5);
    expect(_pooledWidths(scene), {_w});
    restOnly.dispose();
    scene.dispose();
  });

  test('T-WR88c-7 · a view that shrinks at the same scale gets targets of '
      'its own, not a sub-rectangle of the old ones', () async {
    final scene = _scene();
    for (var i = 0; i < 4; i++) {
      await _frame(scene, 1.0);
    }
    for (var i = 0; i < 4; i++) {
      await _frame(scene, 1.0, width: _w / 2);
    }
    expect(
      scene.surface.lastSwapchainExtent(),
      const ui.Size(_w / 2, _h * 1.0),
    );
    expect(scene.surface.lastSwapchainColorTexture()!.width, _w ~/ 2);
    scene.dispose();
  });

  test('T-WR88c-6 · RenderExtent clamps to the last texel center', () {
    const extent = RenderExtent(
      width: 64,
      height: 32,
      targetWidth: 128,
      targetHeight: 64,
    );
    expect(extent.uvMaxX, 63.5 / 128);
    expect(extent.uvMaxY, 31.5 / 64);
    expect(RenderExtent.uvMaxOf(null), (2.0, 2.0));
  });
}
