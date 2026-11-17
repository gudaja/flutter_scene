// WR-81c: the light-to-item assignment must stay byte-identical to the
// original algorithm (full sort of every overflowing item's candidates)
// while (a) picking the nearest maxPerItem lights without a full sort and
// (b) being remembered between frames by LightAssignmentCache when neither
// the lights nor the items changed.
//
// The reference below is a verbatim copy of assignLightsToItems as of fork
// commit ef2fdcab (home3d); random scenes compare its output (index buffer,
// overflow count, every item's slice) against the new code.

import 'dart:math' as math;

import 'package:flutter_scene/scene.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_scene/src/render/bvh.dart';
import 'package:flutter_scene/src/render/light_culling.dart';
import 'package:flutter_scene/src/render/render_scene.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math.dart';

class _StubGeometry extends Geometry {
  @override
  void bind(
    gpu.RenderPass pass,
    TransientWriter transientsBuffer,
    Matrix4 modelTransform,
    Matrix4 cameraTransform,
    Vector3 cameraPosition, {
    gpu.Shader? shaderOverride,
    double depthBias = 0.0,
  }) => throw UnsupportedError('stub');
}

class _StubMaterial extends Material {
  @override
  void bind(
    gpu.RenderPass pass,
    TransientWriter transientsBuffer,
    Lighting l,
  ) => throw UnsupportedError('stub');
}

// ---------------------------------------------------------------------------
// Reference: assignLightsToItems at ef2fdcab, verbatim apart from names.

LightCullResult _referenceAssign({
  required List<RenderItem> items,
  required Bvh bvh,
  required List<CullableLight> lights,
  required int maxPerItem,
}) {
  if (lights.every(
    (light) => light.bounds == null && light.worldPosition == null,
  )) {
    return _refInfinite(items, lights, maxPerItem);
  }
  for (final item in items) {
    item.lightScratch.clear();
  }
  final infinite = <CullableLight>[
    for (final light in lights)
      if (light.bounds == null) light,
  ];
  final unbounded = <RenderItem>[
    for (final item in items)
      if (item.worldBounds == null) item,
  ];
  if (infinite.isNotEmpty) {
    for (final item in items) {
      final mask = item.lightChannelMask;
      for (final light in infinite) {
        if ((light.channelMask & mask) != 0) {
          item.lightScratch.add(light.index);
        }
      }
    }
  }
  for (final light in lights) {
    final bounds = light.bounds;
    if (bounds == null) continue;
    final channelMask = light.channelMask;
    bvh.queryAabb(bounds, (item) {
      if ((item.lightChannelMask & channelMask) != 0) {
        item.lightScratch.add(light.index);
      }
    });
    for (final item in unbounded) {
      if ((item.lightChannelMask & channelMask) != 0) {
        item.lightScratch.add(light.index);
      }
    }
  }
  final flat = <int>[];
  final lightsByIndex = <int, CullableLight>{
    for (final light in lights) light.index: light,
  };
  final slicesByHash = <int, List<int>>{};
  var overflowedItems = 0;
  for (final item in items) {
    final scratch = item.lightScratch;
    var count = scratch.length;
    if (count > maxPerItem) {
      scratch.sort(
        (a, b) => _refCompare(item, lightsByIndex[a]!, lightsByIndex[b]!),
      );
      count = maxPerItem;
      overflowedItems++;
    }
    var hash = count;
    for (var i = 0; i < count; i++) {
      hash = 0x3fffffff & (hash * 31 + scratch[i]);
    }
    final candidates = slicesByHash[hash];
    final shared = candidates == null
        ? -1
        : _refFindSlice(flat, candidates, scratch, count);
    item.lightListCount = count;
    if (shared >= 0) {
      item.lightListOffset = shared;
      continue;
    }
    item.lightListOffset = flat.length;
    (slicesByHash[hash] ??= <int>[]).add(flat.length);
    for (var i = 0; i < count; i++) {
      flat.add(scratch[i]);
    }
  }
  return LightCullResult(flat, overflowedItems);
}

int _refFindSlice(List<int> flat, List<int> offsets, List<int> list, int n) {
  for (final offset in offsets) {
    if (offset + n > flat.length) continue;
    var equal = true;
    for (var i = 0; i < n; i++) {
      if (flat[offset + i] != list[i]) {
        equal = false;
        break;
      }
    }
    if (equal) return offset;
  }
  return -1;
}

LightCullResult _refInfinite(
  List<RenderItem> items,
  List<CullableLight> lights,
  int maxPerItem,
) {
  final flat = <int>[];
  final slices = <int, (int, int, bool)>{};
  var overflowedItems = 0;
  for (final item in items) {
    final mask = item.lightChannelMask;
    var slice = slices[mask];
    if (slice == null) {
      final offset = flat.length;
      var count = 0;
      var capped = false;
      for (final light in lights) {
        if ((light.channelMask & mask) == 0) continue;
        if (count == maxPerItem) {
          capped = true;
          break;
        }
        flat.add(light.index);
        count++;
      }
      slice = (offset, count, capped);
      slices[mask] = slice;
    }
    if (slice.$3) overflowedItems++;
    item.lightListOffset = slice.$1;
    item.lightListCount = slice.$2;
  }
  return LightCullResult(flat, overflowedItems);
}

int _refCompare(RenderItem item, CullableLight a, CullableLight b) {
  final bounds = item.worldBounds;
  final aDistance = _refDistance(item, bounds, a.worldPosition);
  final bDistance = _refDistance(item, bounds, b.worldPosition);
  final distanceOrder = aDistance.compareTo(bDistance);
  if (distanceOrder != 0) return distanceOrder;
  if (bounds != null && a.worldPosition != null && b.worldPosition != null) {
    final cx = (bounds.min.x + bounds.max.x) * 0.5;
    final cy = (bounds.min.y + bounds.max.y) * 0.5;
    final cz = (bounds.min.z + bounds.max.z) * 0.5;
    final ac = _refDistanceSquared(a.worldPosition!, cx, cy, cz);
    final bc = _refDistanceSquared(b.worldPosition!, cx, cy, cz);
    final centerOrder = ac.compareTo(bc);
    if (centerOrder != 0) return centerOrder;
  }
  return a.index.compareTo(b.index);
}

double _refDistance(RenderItem item, Aabb3? bounds, Vector3? position) {
  if (position == null) return double.negativeInfinity;
  if (bounds == null) {
    final t = item.worldTransform.storage;
    return _refDistanceSquared(position, t[12], t[13], t[14]);
  }
  final dx = position.x < bounds.min.x
      ? bounds.min.x - position.x
      : position.x > bounds.max.x
      ? position.x - bounds.max.x
      : 0.0;
  final dy = position.y < bounds.min.y
      ? bounds.min.y - position.y
      : position.y > bounds.max.y
      ? position.y - bounds.max.y
      : 0.0;
  final dz = position.z < bounds.min.z
      ? bounds.min.z - position.z
      : position.z > bounds.max.z
      ? position.z - bounds.max.z
      : 0.0;
  return dx * dx + dy * dy + dz * dz;
}

double _refDistanceSquared(Vector3 p, double x, double y, double z) {
  final dx = p.x - x;
  final dy = p.y - y;
  final dz = p.z - z;
  return dx * dx + dy * dy + dz * dz;
}

// ---------------------------------------------------------------------------
// Random scenes.

class _Scene {
  _Scene(this.items, this.lights);
  final List<RenderItem> items;
  List<CullableLight> lights;
  Bvh bvh = Bvh.build(const []);

  void rebuildBvh() {
    bvh = Bvh.build([
      for (final item in items)
        if (item.frustumCulled && item.worldBounds != null) item,
    ]);
  }
}

const _masks = [0xFF, 0xFF, 0xFF, 0x01, 0x02, 0x06, 0x80];

// Coordinates snap to a coarse grid so equal distances (and the center and
// index tie-breaks) actually occur.
double _snap(math.Random r, double extent) =>
    (r.nextDouble() * extent * 4).roundToDouble() / 4;

RenderItem _randomItem(math.Random r, {required bool channels}) {
  final item = RenderItem(geometry: _StubGeometry(), material: _StubMaterial());
  final kind = r.nextInt(20);
  if (kind == 0) {
    // Unbounded: distance measured to its transform's translation.
    item.worldTransform.setTranslationRaw(
      _snap(r, 40),
      _snap(r, 6),
      _snap(r, 40),
    );
  } else {
    final x = _snap(r, 40), y = _snap(r, 6), z = _snap(r, 40);
    // A few level-spanning items (merged walls, floors).
    final big = kind == 1 || kind == 2;
    final sx = big ? 20 + _snap(r, 20) : 0.25 + _snap(r, 2);
    final sy = big ? 3.0 : 0.25 + _snap(r, 2);
    final sz = big ? 20 + _snap(r, 20) : 0.25 + _snap(r, 2);
    item.worldBounds = Aabb3.minMax(
      Vector3(x, y, z),
      Vector3(x + sx, y + sy, z + sz),
    );
    // Not frustum-culled: outside the BVH, so it gets no ranged light.
    if (kind == 3) item.frustumCulled = false;
  }
  if (channels) item.lightChannelMask = _masks[r.nextInt(_masks.length)];
  return item;
}

CullableLight _randomLight(math.Random r, int index, {required bool channels}) {
  final mask = channels ? _masks[r.nextInt(_masks.length)] : 0xFF;
  final kind = r.nextInt(30);
  if (kind == 0) return CullableLight(index, null, channelMask: mask);
  final position = Vector3(_snap(r, 40), _snap(r, 6), _snap(r, 40));
  // Range 0 = positioned light with infinite influence.
  final range = kind == 1 ? 0.0 : 1 + _snap(r, 8);
  return CullableLight(
    index,
    lightInfluenceBounds(position, range),
    worldPosition: position,
    channelMask: mask,
  );
}

_Scene _randomScene(math.Random r, {int? lightCount, bool? channels}) {
  final useChannels = channels ?? r.nextBool();
  final items = [
    for (var i = 0, n = 5 + r.nextInt(120); i < n; i++)
      _randomItem(r, channels: useChannels),
  ];
  final count = lightCount ?? r.nextInt(160);
  final lights = [
    for (var i = 0; i < count; i++) _randomLight(r, i, channels: useChannels),
  ];
  // Rows are not always visited in index order.
  if (r.nextInt(4) == 0) lights.shuffle(r);
  return _Scene(items, lights)..rebuildBvh();
}

typedef _Snapshot = ({
  List<int> indices,
  int overflow,
  List<(int, int)> slices,
});

_Snapshot _capture(List<RenderItem> items, LightCullResult result) => (
  indices: List.of(result.indices),
  overflow: result.overflowedItemCount,
  slices: [for (final i in items) (i.lightListOffset, i.lightListCount)],
);

_Snapshot _reference(_Scene s, int maxPerItem) => _capture(
  s.items,
  _referenceAssign(
    items: s.items,
    bvh: s.bvh,
    lights: s.lights,
    maxPerItem: maxPerItem,
  ),
);

void _expectSame(_Snapshot actual, _Snapshot expected, String why) {
  expect(actual.indices, expected.indices, reason: '$why: indices');
  expect(actual.overflow, expected.overflow, reason: '$why: overflow');
  expect(actual.slices, expected.slices, reason: '$why: item slices');
}

void main() {
  test('T-WR81c-1 · top-k selection matches the full-sort reference', () {
    final r = math.Random(81);
    var overflowing = 0;
    for (var round = 0; round < 300; round++) {
      final s = _randomScene(r);
      final maxPerItem = const [1, 4, 16, 16, 16][r.nextInt(5)];
      final expected = _reference(s, maxPerItem);
      overflowing += expected.overflow;
      final actual = _capture(
        s.items,
        assignLightsToItems(
          items: s.items,
          bvh: s.bvh,
          lights: s.lights,
          maxPerItem: maxPerItem,
        ),
      );
      _expectSame(actual, expected, 'round $round');
    }
    // The scenes must actually exercise the capped path.
    expect(overflowing, greaterThan(1000));
  });

  test('T-WR81c-2 · the cache reuses an unchanged assignment', () {
    final r = math.Random(7);
    final s = _randomScene(r, lightCount: 120, channels: false);
    final cache = LightAssignmentCache();
    final first = cache.assign(
      items: s.items,
      bvh: s.bvh,
      lights: s.lights,
      maxPerItem: 16,
    );
    expect(cache.recomputeCount, 1);
    // A new frame repacks the lights into fresh objects with equal values.
    final repacked = [
      for (final l in s.lights)
        CullableLight(
          l.index,
          l.bounds == null ? null : Aabb3.copy(l.bounds!),
          worldPosition: l.worldPosition?.clone(),
          channelMask: l.channelMask,
        ),
    ];
    final second = cache.assign(
      items: s.items,
      bvh: s.bvh,
      lights: repacked,
      maxPerItem: 16,
    );
    expect(cache.recomputeCount, 1);
    expect(identical(second, first), isTrue);
    s.lights = repacked;
    _expectSame(_capture(s.items, second), _reference(s, 16), 'reused');
  });

  test('T-WR81c-3 · every change invalidates the cache and matches the '
      'reference', () {
    final r = math.Random(1981);
    for (var round = 0; round < 40; round++) {
      final s = _randomScene(r, lightCount: 40 + r.nextInt(120));
      final cache = LightAssignmentCache();
      var expectedRecomputes = 0;
      for (var step = 0; step < 25; step++) {
        final change = r.nextInt(9);
        var changed = true;
        switch (change) {
          case 0: // Nothing changed.
            changed = step == 0;
          case 1: // A light moved.
            final i = r.nextInt(math.max(s.lights.length, 1));
            if (s.lights.isEmpty) {
              changed = step == 0;
              break;
            }
            s.lights = List.of(s.lights)
              ..[i] = _randomLight(r, s.lights[i].index, channels: true);
          case 2: // A light was added.
            s.lights = [
              ...s.lights,
              _randomLight(r, s.lights.length, channels: false),
            ];
          case 3: // A light was removed.
            if (s.lights.isEmpty) {
              changed = step == 0;
              break;
            }
            s.lights = List.of(s.lights)..removeLast();
          case 4: // An item moved and the BVH was refit.
            final item = s.items[r.nextInt(s.items.length)];
            final b = item.worldBounds;
            if (b == null) {
              item.worldTransform.setTranslationRaw(
                _snap(r, 40),
                1,
                _snap(r, 40),
              );
            } else {
              final d = Vector3(_snap(r, 4) - 2, 0, _snap(r, 4) - 2);
              b.copyFrom(Aabb3.minMax(b.min + d, b.max + d));
              s.bvh.refit();
            }
          case 5: // An item's light channels changed.
            final item = s.items[r.nextInt(s.items.length)];
            item.lightChannelMask = item.lightChannelMask == 0xFF ? 0x01 : 0xFF;
          case 6: // An item was added (structural rebuild).
            s.items.add(_randomItem(r, channels: false));
            s.rebuildBvh();
          case 7: // An item left the BVH (frustum culling switched off).
            final item = s.items[r.nextInt(s.items.length)];
            item.frustumCulled = !item.frustumCulled;
            s.rebuildBvh();
          case 8: // Two items swapped places in the item list.
            final a = r.nextInt(s.items.length);
            final b = r.nextInt(s.items.length);
            if (a == b) {
              changed = step == 0;
              break;
            }
            final t = s.items[a];
            s.items[a] = s.items[b];
            s.items[b] = t;
        }
        if (changed) expectedRecomputes++;
        final result = cache.assign(
          items: s.items,
          bvh: s.bvh,
          lights: s.lights,
          maxPerItem: 16,
        );
        final actual = _capture(s.items, result);
        _expectSame(
          actual,
          _reference(s, 16),
          'round $round step $step change $change',
        );
        expect(
          cache.recomputeCount,
          expectedRecomputes,
          reason: 'round $round step $step change $change',
        );
      }
    }
  });

  test('T-WR81c-5 · a refit that lands a frame after the bounds changed '
      'still recomputes', () {
    final r = math.Random(5);
    final s = _randomScene(r, lightCount: 100, channels: false);
    final cache = LightAssignmentCache();
    cache.assign(items: s.items, bvh: s.bvh, lights: s.lights, maxPerItem: 16);
    // Move every bounded item far away without refitting: the stale tree
    // still places them where they were.
    for (final item in s.items) {
      final b = item.worldBounds;
      if (b == null) continue;
      final d = Vector3(500, 0, 500);
      b.copyFrom(Aabb3.minMax(b.min + d, b.max + d));
    }
    var result = cache.assign(
      items: s.items,
      bvh: s.bvh,
      lights: s.lights,
      maxPerItem: 16,
    );
    _expectSame(_capture(s.items, result), _reference(s, 16), 'stale tree');
    // The refit arrives next frame; the items' bounds did not change again.
    s.bvh.refit();
    result = cache.assign(
      items: s.items,
      bvh: s.bvh,
      lights: s.lights,
      maxPerItem: 16,
    );
    _expectSame(_capture(s.items, result), _reference(s, 16), 'refit');
    expect(cache.recomputeCount, 3);
  });

  test('T-WR81c-4 · a cache shared by two light sets (environment capture '
      'between frames) never hands back the other set', () {
    final r = math.Random(4);
    final s = _randomScene(r, lightCount: 80, channels: false);
    final fewer = s.lights.sublist(0, 30);
    final cache = LightAssignmentCache();
    for (var i = 0; i < 4; i++) {
      final lights = i.isEven ? s.lights : fewer;
      final result = cache.assign(
        items: s.items,
        bvh: s.bvh,
        lights: lights,
        maxPerItem: 16,
      );
      final actual = _capture(s.items, result);
      _expectSame(
        actual,
        _reference(_Scene(s.items, lights)..bvh = s.bvh, 16),
        'pass $i',
      );
    }
    expect(cache.recomputeCount, 4);
  });
}
