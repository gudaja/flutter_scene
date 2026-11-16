import 'package:flutter_test/flutter_test.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/light.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/render/shadow_cache.dart';
import 'package:flutter_scene/scene.dart' show PerspectiveCamera;
import 'package:vector_math/vector_math.dart';

// WR-77: site shadow map (DirectionalLight.shadowSiteBounds).
void main() {
  final bounds = Aabb3.minMax(Vector3(-12, -1, -8), Vector3(18, 9, 22));

  Iterable<Vector3> corners(Aabb3 b) sync* {
    for (var i = 0; i < 8; i++) {
      yield Vector3(
        (i & 1) == 0 ? b.min.x : b.max.x,
        (i & 2) == 0 ? b.min.y : b.max.y,
        (i & 4) == 0 ? b.min.z : b.max.z,
      );
    }
  }

  for (final dir in [
    Vector3(-0.3, -1.0, -0.2),
    Vector3(0.5, -0.4, 0.7),
    Vector3(0.0, -1.0, 0.0),
    Vector3(-0.9, -0.25, 0.1),
  ]) {
    test('site cascade fits every corner inside the tile, dir $dir', () {
      final light = DirectionalLight(
        direction: dir,
        castsShadow: true,
        shadowMapResolution: 2048,
      )..shadowSiteBounds = bounds;
      final cascades = light.computeCascades(
        PerspectiveCamera(position: Vector3(40, 20, 40)),
        1.5,
      );
      expect(cascades, hasLength(1));
      final c = cascades.single;
      var maxX = 0.0, maxY = 0.0;
      for (final p in corners(bounds)) {
        final clip = c.lightSpaceMatrix.transformed(Vector4(p.x, p.y, p.z, 1));
        expect(clip.w, closeTo(1.0, 1e-9));
        expect(clip.x.abs(), lessThan(1.0));
        expect(clip.y.abs(), lessThan(1.0));
        expect(clip.z, inInclusiveRange(0.0, 1.0));
        maxX = clip.x.abs() > maxX ? clip.x.abs() : maxX;
        maxY = clip.y.abs() > maxY ? clip.y.abs() : maxY;
      }
      // Tight: the farthest corner sits within the one-texel margin.
      expect(maxX, greaterThan(0.99));
      expect(maxY, greaterThan(0.99));
      // Toward the sun is nearer (smaller depth).
      final n = dir.normalized();
      final a = c.lightSpaceMatrix.transformed(Vector4(0, 0, 0, 1));
      final b = c.lightSpaceMatrix.transformed(
        Vector4(-n.x * 3, -n.y * 3, -n.z * 3, 1),
      );
      expect(b.z, lessThan(a.z));
      // Depth range is 7 * boxSize, as the shader's bias conversion assumes.
      expect((b.z - a.z).abs() * 7.0 * c.boxSize, closeTo(3.0, 1e-6));
    });
  }

  test('camera moves do not change the site cascade or refresh its tile', () {
    final light = DirectionalLight(
      direction: Vector3(-0.3, -1.0, -0.2),
      castsShadow: true,
    )..shadowSiteBounds = bounds;
    final cache = DirectionalShadowCache();
    final first = cache.plan(
      light: light,
      lightDirection: light.direction,
      idealCascades: light.computeCascades(
        PerspectiveCamera(position: Vector3(40, 20, 40)),
        1.5,
      ),
      staticSignature: 7,
    );
    expect(first.refreshes, hasLength(1));
    final second = cache.plan(
      light: light,
      lightDirection: light.direction,
      idealCascades: light.computeCascades(
        PerspectiveCamera(
          position: Vector3(-30, 5, 10),
          target: Vector3(3, 0, 3),
        ),
        0.5,
      ),
      staticSignature: 7,
    );
    expect(second.refreshes, isEmpty);
    expect(
      second.cascades.single.lightSpaceMatrix,
      first.cascades.single.lightSpaceMatrix,
    );
    // A sun step refreshes it.
    light.direction = Vector3(-0.35, -1.0, -0.2);
    final third = cache.plan(
      light: light,
      lightDirection: light.direction,
      idealCascades: light.computeCascades(
        PerspectiveCamera(position: Vector3(40, 20, 40)),
        1.5,
      ),
      staticSignature: 7,
    );
    expect(third.refreshes, hasLength(1));
  });
}
