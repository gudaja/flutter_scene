import 'dart:math' as math;

// ignore: implementation_imports
import 'package:flutter_scene/src/camera.dart';
import 'package:flutter_test/flutter_test.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/light.dart';
// ignore: implementation_imports
import 'package:flutter_scene/src/render/shadow_cache.dart';
import 'package:vector_math/vector_math.dart';

ShadowCascade cascade(Vector3 center, double radius, double split) =>
    ShadowCascade(
      lightSpaceMatrix: Matrix4.identity(),
      splitDistance: split,
      boxSize: radius * 2.0,
      center: center,
      radius: radius,
    );

List<ShadowCascade> idealCascades({Vector3? offset}) {
  final o = offset ?? Vector3.zero();
  return [
    cascade(Vector3(0, 0, 5) + o, 6.0, 10.0),
    cascade(Vector3(0, 0, 20) + o, 25.0, 45.0),
  ];
}

void main() {
  late DirectionalLight light;
  late DirectionalShadowCache cache;

  setUp(() {
    light = DirectionalLight()
      ..direction = Vector3(0.3, -1.0, 0.2).normalized()
      ..shadowMapResolution = 512;
    cache = DirectionalShadowCache();
  });

  ShadowCachePlan plan(List<ShadowCascade> ideal, {int signature = 1}) =>
      cache.plan(
        light: light,
        lightDirection: light.direction,
        idealCascades: ideal,
        staticSignature: signature,
      );

  test('first frame refreshes every cascade', () {
    final p = plan(idealCascades());
    expect(p.refreshes.length, 2);
    expect(p.cascades.length, 2);
    // Effective boxes carry the slack.
    expect(
      p.cascades[0].boxSize,
      closeTo(6.0 * DirectionalShadowCache.slackFactor * 2.0, 1e-9),
    );
  });

  test('a stable view refreshes nothing and keeps the same matrices', () {
    plan(idealCascades());
    final p = plan(idealCascades());
    expect(p.refreshes, isEmpty);
    final q = plan(idealCascades());
    expect(q.cascades[1].lightSpaceMatrix, p.cascades[1].lightSpaceMatrix);
  });

  test('drift within the slack keeps the cached tiles', () {
    plan(idealCascades());
    // 10% of the small cascade's radius, under the 15% slack.
    final p = plan(idealCascades(offset: Vector3(0.6, 0, 0)));
    expect(p.refreshes, isEmpty);
  });

  test('drift past the slack re-renders the cascade that no longer fits', () {
    plan(idealCascades());
    // 20% of the near cascade's radius (over the slack), but only 4.8% of
    // the far cascade's.
    final p = plan(idealCascades(offset: Vector3(1.2, 0, 0)));
    expect(p.refreshes.length, 1);
    expect(p.refreshes.single.cascadeIndex, 0);
  });

  test('content changes refresh amortized, nearest cascade first', () {
    plan(idealCascades());
    final p1 = plan(idealCascades(), signature: 2);
    expect(p1.refreshes.length, DirectionalShadowCache.maxAmortizedRefreshes);
    expect(p1.refreshes.first.cascadeIndex, 0);
    final p2 = plan(idealCascades(), signature: 2);
    expect(p2.refreshes.length, 1);
    expect(p2.refreshes.first.cascadeIndex, 1);
    final p3 = plan(idealCascades(), signature: 2);
    expect(p3.refreshes, isEmpty);
  });

  test('a small light turn refreshes amortized, nearest cascade first', () {
    plan(idealCascades());
    // A 2 degree step, inside the lag tolerance.
    light.direction = Quaternion.axisAngle(
      Vector3(0, 0, 1),
      2.0 * degrees2Radians,
    ).rotated(light.direction);
    final p1 = plan(idealCascades());
    expect(p1.refreshes.length, DirectionalShadowCache.maxAmortizedRefreshes);
    expect(p1.refreshes.single.cascadeIndex, 0);
    // The far cascade still samples through its old matrix.
    final oldFar = Matrix4.copy(p1.cascades[1].lightSpaceMatrix);
    final p2 = plan(idealCascades());
    expect(p2.refreshes.single.cascadeIndex, 1);
    expect(p2.cascades[1].lightSpaceMatrix, isNot(oldFar));
    expect(plan(idealCascades()).refreshes, isEmpty);
  });

  test('a large light turn re-renders everything', () {
    plan(idealCascades());
    light.direction = Vector3(-0.5, -1.0, 0.1).normalized();
    final p = plan(idealCascades());
    expect(p.refreshes.length, 2);
  });

  test('a resolution change rebuilds everything', () {
    plan(idealCascades());
    light.shadowMapResolution = 1024;
    final p = plan(idealCascades());
    expect(p.refreshes.length, 2);
  });

  test('drift along the light keeps the cached tiles', () {
    plan(idealCascades());
    // Half the near cascade's radius toward the light, far past the slack
    // but inside the tile's deep depth range.
    final along = light.direction.normalized() * 3.0;
    final p = plan(idealCascades(offset: along));
    expect(p.refreshes, isEmpty);
    expect(
      DirectionalShadowCache.sphereInsideTile(
        p.cascades[0].lightSpaceMatrix,
        Vector3(0, 0, 5) + along,
        6.0,
      ),
      isTrue,
    );
  });

  test('a tile is usable exactly while the ideal sphere lies in its box', () {
    final m = light.cascadeLightSpaceMatrix(
      light.direction.normalized(),
      Vector3.zero(),
      6.0 * DirectionalShadowCache.slackFactor,
    );
    // Across the light: the box's half side is the slack radius.
    final dir = light.direction.normalized();
    final across = Vector3(dir.z, 0, -dir.x).normalized();
    final room = 6.0 * (DirectionalShadowCache.slackFactor - 1.0);
    bool inside(Vector3 c) =>
        DirectionalShadowCache.sphereInsideTile(m, c, 6.0);
    expect(inside(Vector3.zero()), isTrue);
    expect(inside(across * (room * 0.99)), isTrue);
    expect(inside(across * (room * 1.01)), isFalse);
    // Away from the light the depth range ends 2 slack radii past the center.
    final depthRoom = 2.0 * 6.0 * DirectionalShadowCache.slackFactor - 6.0;
    expect(inside(dir * (depthRoom * 0.99)), isTrue);
    expect(inside(dir * (depthRoom * 1.01)), isFalse);
  });

  test('steady drift across the light leads the re-rendered tile', () {
    final dir = light.direction.normalized();
    final across = Vector3(dir.z, 0, -dir.x).normalized();
    plan(idealCascades());
    final room = 6.0 * (DirectionalShadowCache.slackFactor - 1.0);
    const steps = 400;
    final step = room * 0.05;
    var nearRefreshes = 0;
    for (var i = 1; i <= steps; i++) {
      final offset = across * (step * i);
      final p = plan(idealCascades(offset: offset));
      nearRefreshes += p.refreshes.where((r) => r.cascadeIndex == 0).length;
      // Coverage never lapses: the sphere every fragment of the cascade lies
      // in stays inside the tile it samples.
      expect(
        DirectionalShadowCache.sphereInsideTile(
          p.cascades[0].lightSpaceMatrix,
          Vector3(0, 0, 5) + offset,
          6.0,
        ),
        isTrue,
        reason: 'step $i',
      );
    }
    // A centered tile refreshes every `room` of travel (20 refreshes over
    // 20 rooms); a led one crosses 1 + leadFraction rooms per refresh.
    final travel = step * steps / room;
    expect(
      nearRefreshes,
      lessThanOrEqualTo(
        (travel / (1.0 + DirectionalShadowCache.leadFraction)).ceil() + 1,
      ),
    );
  });

  test(
    'an orbiting camera keeps every cascade covered with fewer refreshes',
    () {
      light
        ..direction = Vector3(-0.4, -0.7, -0.5).normalized()
        ..shadowMapResolution = 2048
        ..shadowCascadeCount = 3
        ..shadowMaxDistance = 60.0
        ..firstCascadeFarBound = 12.0;
      const distance = 20.0;
      var refreshes = 0;
      const frames = 360;
      for (var f = 0; f <= frames; f++) {
        final yaw = f * degrees2Radians;
        final camera = PerspectiveCamera(
          position: Vector3(
            distance * 0.8 * math.cos(yaw),
            distance * 0.6,
            distance * 0.8 * math.sin(yaw),
          ),
          target: Vector3.zero(),
        );
        final ideal = light.computeCascades(camera, 16 / 9);
        final p = plan(ideal);
        if (f > 0) refreshes += p.refreshes.length;
        for (var c = 0; c < ideal.length; c++) {
          expect(
            DirectionalShadowCache.sphereInsideTile(
              p.cascades[c].lightSpaceMatrix,
              ideal[c].center!,
              ideal[c].radius,
            ),
            isTrue,
            reason: 'frame $f cascade $c',
          );
        }
      }
      // The centered-tile rule with a 3D-distance test refreshed 68 tiles over
      // this orbit with the same slack; the exact box test alone gives 54, and
      // leading the re-rendered tile 35.
      expect(refreshes, lessThanOrEqualTo(40));
    },
  );

  test('a shadow-caster channel change rebuilds everything', () {
    plan(idealCascades());
    expect(plan(idealCascades()).refreshes, isEmpty);
    light.shadowCasterChannelMask = 0x0F;
    expect(plan(idealCascades()).refreshes.length, 2);
  });
}
