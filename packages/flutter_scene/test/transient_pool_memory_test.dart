// WR-88b (home3d): what the transient texture pools keep. The shadow atlas
// lives in one size-independent pool per view (a render scale that toggles
// keeps one copy), a ring slot is reused once the GPU finished the frame that
// last used it, no atlas is allocated without a caster, cached cascades
// without dynamic casters compose without an atlas-sized depth, spot and
// point tiles can keep their own resolution, and an atlas shape the frame
// stopped using is released. GPU-gated; acquiring a texture needs a device.

// ignore_for_file: invalid_use_of_internal_member, implementation_imports

import 'dart:ui' as ui;

import 'package:flutter_scene/scene.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_scene/src/render/frame_transients.dart';
import 'package:flutter_scene/src/render/render_graph.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math.dart';

import 'support/gpu_available.dart';

const _atlasName = 'directional_shadow_map';
const _atlasDepthName = 'directional_shadow_map_depth';

TransientTextureDescriptor _color(int w, int h, [String name = 'c']) =>
    TransientTextureDescriptor.color(
      width: w,
      height: h,
      format: gpu.PixelFormat.r8g8b8a8UNormInt,
      debugName: name,
    );

/// Renders [scene] once at [width] x [height] pixels and waits for the GPU
/// work to report completion, as a paced app frame would.
Future<void> _frame(Scene scene, {int width = 64, int height = 32}) async {
  final recorder = ui.PictureRecorder();
  try {
    scene.renderViews(
      [
        RenderView(
          camera: PerspectiveCamera(
            position: Vector3(0, 4, 8),
            target: Vector3.zero(),
          ),
        ),
      ],
      ui.Canvas(recorder),
      region: ui.Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
      pixelRatio: 1.0,
    );
  } finally {
    recorder.endRecording().dispose();
  }
  // Every submission, not only the frame's last: uploads trail the frame.
  for (
    var i = 0;
    i < 100 &&
        rendererSubmissions.completedThrough <
            rendererSubmissions.latestSubmission;
    i++
  ) {
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
}

Node _sun({int resolution = 256, int cascades = 3}) => Node()
  ..addComponent(
    DirectionalLightComponent(
      DirectionalLight(
        castsShadow: true,
        shadowMapResolution: resolution,
        shadowCascadeCount: cascades,
      ),
    ),
  );

Node _caster({bool shadowStatic = false}) =>
    Node(mesh: Mesh(CuboidGeometry(Vector3.all(1)), PhysicallyBasedMaterial()))
      ..shadowStatic = shadowStatic;

Node _ground() => Node(
  mesh: Mesh(PlaneGeometry(width: 10, depth: 10), PhysicallyBasedMaterial()),
)..position = Vector3(0, -0.5, 0);

Node _lamp(Vector3 at) => Node()
  ..position = at
  ..addComponent(
    PointLightComponent(
      PointLight(range: 10.0, castsShadow: true, shadowMapResolution: 32),
    ),
  );

/// Every descriptor the view's pools hold, with its live ring slots: the
/// size-independent pool first, then the current size's.
Map<TransientTextureDescriptor, int> _held(Scene scene) {
  final pool = scene.surface.transientTexturePool();
  return {...pool.sizeIndependent.debugHeld, ...pool.debugHeld};
}

Iterable<TransientTextureDescriptor> _named(Scene scene, String name) =>
    _held(scene).keys.where((d) => d.debugName == name);

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  if (!gpuAvailable()) {
    test(
      'transient pool memory suite (skipped: no GPU device)',
      () {},
      skip: 'Requires a GPU device.',
    );
    return;
  }

  setUpAll(Scene.initializeStaticResources);

  group('ring slots', () {
    test('T-WR88b-1 · a frame whose GPU work completed is reused', () {
      final pool = TransientTexturePool();
      pool.beginFrame();
      final first = pool.acquire(_color(8, 8));
      rendererSubmissions.complete(rendererSubmissions.record());
      pool.beginFrame();
      expect(pool.acquire(_color(8, 8)), same(first));
      expect(pool.heldTextureCount, 1);
    });

    test('T-WR88b-2 · a frame still in flight rotates; the idle slot is '
        'released', () {
      final pool = TransientTexturePool();
      pool.beginFrame();
      final first = pool.acquire(_color(8, 8));
      final pending = rendererSubmissions.record();
      pool.beginFrame();
      final second = pool.acquire(_color(8, 8));
      expect(second, isNot(same(first)));
      expect(pool.heldTextureCount, 2);
      rendererSubmissions.complete(pending);
      for (var i = 0; i <= TransientTexturePool.kRingSlotIdleFrames + 1; i++) {
        pool.beginFrame();
        pool.acquire(_color(8, 8));
      }
      expect(pool.heldTextureCount, 1);
    });

    test('T-WR88b-3 · a superseded shape is released once its frame '
        'completed', () {
      final pool = TransientTexturePool(releaseUnusedShapes: true);
      pool.beginFrame();
      pool.acquire(_color(16, 8, 'atlas'));
      final pending = rendererSubmissions.record();
      pool.beginFrame();
      pool.acquire(_color(32, 8, 'atlas'));
      pool.beginFrame();
      pool.acquire(_color(32, 8, 'atlas'));
      // The old shape's last frame is still running.
      expect(pool.debugHeld.keys.map((d) => d.width), contains(16));
      rendererSubmissions.complete(pending);
      pool.beginFrame();
      pool.acquire(_color(32, 8, 'atlas'));
      expect(pool.debugHeld.keys.map((d) => d.width), [32]);
    });
  });

  group('shadow atlas', () {
    test('T-WR88b-4 · cascades: one atlas for both render scales, kept in '
        'the size-independent pool', () async {
      final scene = Scene()
        ..add(_sun())
        ..add(_caster())
        ..add(_ground());
      await _frame(scene);
      await _frame(scene, width: 32, height: 16);
      await _frame(scene);
      final pool = scene.surface.transientTexturePool();
      final atlases = pool.sizeIndependent.debugHeld.keys.where(
        (d) => d.debugName == _atlasName,
      );
      expect(atlases.map((d) => (d.width, d.height)), [(3 * 256, 256)]);
      expect(
        pool.debugHeld.keys.where((d) => d.debugName == _atlasName),
        isEmpty,
      );
      expect(
        pool.sizeIndependent.debugHeld[atlases.single],
        1,
        reason: 'paced frames keep one ring slot',
      );
      scene.dispose();
    });

    test('T-WR88b-5 · no caster: no atlas and no cached tiles', () async {
      final scene = Scene()
        ..add(_sun())
        ..add(_ground()..castsShadows = false);
      await _frame(scene);
      await _frame(scene);
      expect(_named(scene, _atlasName), isEmpty);
      expect(_named(scene, _atlasDepthName), isEmpty);
      // A caster appearing brings the atlas back.
      scene.add(_caster());
      await _frame(scene);
      expect(_named(scene, _atlasName), hasLength(1));
      scene.dispose();
    });

    test('T-WR88b-6 · cached cascades without a dynamic caster compose '
        'without an atlas-sized depth', () async {
      final scene = Scene()
        ..add(_sun())
        ..add(_caster(shadowStatic: true))
        ..add(_ground()..shadowStatic = true);
      await _frame(scene);
      await _frame(scene);
      expect(_named(scene, _atlasName), hasLength(1));
      expect(_named(scene, _atlasDepthName), isEmpty);
      scene.dispose();
    });

    test('T-WR88b-7 · lamps at their own resolution; switching them off '
        'releases the wider atlas', () async {
      final scene = Scene()
        ..punctualShadowsAtOwnResolution = true
        ..add(_sun())
        ..add(_caster())
        ..add(_ground());
      final lamps = [_lamp(Vector3(1, 2, 0)), _lamp(Vector3(-1, 2, 0))];
      lamps.forEach(scene.add);
      await _frame(scene);
      await _frame(scene);
      // Two point lights = four 64 texel tiles, four to a 256 column.
      expect(_named(scene, _atlasName).map((d) => d.width), [3 * 256 + 64]);
      lamps.forEach(scene.remove);
      await _frame(scene);
      await _frame(scene);
      expect(_named(scene, _atlasName).map((d) => d.width), [3 * 256]);
      scene.dispose();
    });

    test('T-WR88b-8 · without the switch lamp tiles keep the cascade '
        'resolution', () async {
      final scene = Scene()
        ..add(_sun())
        ..add(_caster())
        ..add(_ground())
        ..add(_lamp(Vector3(1, 2, 0)));
      await _frame(scene);
      expect(_named(scene, _atlasName).map((d) => d.width), [5 * 256]);
      scene.dispose();
    });
  });

  test('T-WR88b-9 · punctualShadowTileResolution', () {
    expect(punctualShadowTileResolution(2048, 1024), 1024);
    expect(punctualShadowTileResolution(2048, 512), 512);
    expect(punctualShadowTileResolution(2048, 2048), 2048);
    expect(punctualShadowTileResolution(1024, 2048), 1024);
    expect(punctualShadowTileResolution(2048, 768), 2048);
    expect(punctualShadowTileResolution(1536, 512), 1536);
  });
}
