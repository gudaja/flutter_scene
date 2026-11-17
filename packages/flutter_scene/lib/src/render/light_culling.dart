import 'dart:typed_data';

import 'package:vector_math/vector_math.dart';

import 'package:flutter_scene/src/render/bvh.dart';
import 'package:flutter_scene/src/render/render_scene.dart';

/// A punctual light reduced to what culling needs: its row [index] in the
/// shared light-parameter buffer, world-space influence [bounds], the
/// [channelMask] it lights, and optional [worldPosition] used to choose the
/// nearest lights when an item overflows its light budget.
///
/// A null [bounds] means infinite influence (a directional light, or a point or
/// spot light with no range), so the light reaches every item and is never
/// culled. A null [worldPosition] identifies a directional light, which sorts
/// ahead of local lights because its influence is independent of distance.
class CullableLight {
  const CullableLight(
    this.index,
    this.bounds, {
    this.worldPosition,
    this.channelMask = 0xFF,
  });

  final int index;
  final Aabb3? bounds;
  final Vector3? worldPosition;

  /// The light's channel mask; it reaches an item only when this intersects
  /// [RenderItem.lightChannelMask].
  final int channelMask;
}

/// The world-space AABB a point or spot light at [worldPosition] with [range]
/// can influence, or null when [range] is not positive (infinite influence).
///
/// A spot light is bounded by the same sphere as a point light of its range; a
/// tighter cone bound is a future refinement.
Aabb3? lightInfluenceBounds(Vector3 worldPosition, double range) {
  if (range <= 0.0) return null;
  final r = Vector3.all(range);
  return Aabb3.minMax(worldPosition - r, worldPosition + r);
}

/// The result of a cull: the flattened light-index buffer (each item's slice is
/// `[offset, offset + count)`, written onto the items) and how many items had
/// more lights than [maxPerItem] and dropped the excess.
class LightCullResult {
  const LightCullResult(this.indices, this.overflowedItemCount);

  final List<int> indices;

  /// The number of items whose light list was capped this cull.
  final int overflowedItemCount;

  /// Whether any item dropped lights.
  bool get overflowed => overflowedItemCount > 0;
}

/// Assigns each item in [items] the punctual [lights] that reach it, writing
/// [RenderItem.lightListOffset]/[RenderItem.lightListCount] onto each and
/// returning the flattened index buffer the light-index texture is built from.
///
/// Infinite-influence lights reach every item. Ranged lights are scattered onto
/// the items their influence AABB overlaps using [bvh] (which holds the bounded
/// items); unbounded items (no [RenderItem.worldBounds]) cannot be culled, so
/// they conservatively receive every light. A light is skipped for an item
/// whose [RenderItem.lightChannelMask] its own [CullableLight.channelMask] does
/// not intersect, whatever the geometry says. Each item's list is capped at
/// [maxPerItem]. Directional lights and the nearest local lights are retained;
/// the excess is dropped and flagged in the result.
LightCullResult assignLightsToItems({
  required List<RenderItem> items,
  required Bvh bvh,
  required List<CullableLight> lights,
  required int maxPerItem,
}) {
  if (lights.every(
    (light) => light.bounds == null && light.worldPosition == null,
  )) {
    return _assignInfiniteLights(items, lights, maxPerItem);
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

  // Infinite-influence lights reach every item whose channels they share.
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

  // Ranged lights: scatter onto bounded items via the BVH, and onto the (few)
  // unbounded items unconditionally since they cannot be culled.
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
  // Items with identical lists share one slice, so instancing can merge them
  // (batches require equal slices).
  final slicesByHash = <int, List<int>>{};
  var overflowedItems = 0;
  for (final item in items) {
    final scratch = item.lightScratch;
    var count = scratch.length;
    if (count > maxPerItem) {
      _keepNearest(item, scratch, lightsByIndex, maxPerItem);
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
        : _findSlice(flat, candidates, scratch, count);
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

// The offset of the slice in [flat] at one of [offsets] that holds the first
// [count] entries of [list], or -1.
int _findSlice(List<int> flat, List<int> offsets, List<int> list, int count) {
  for (final offset in offsets) {
    if (offset + count > flat.length) continue;
    var equal = true;
    for (var i = 0; i < count; i++) {
      if (flat[offset + i] != list[i]) {
        equal = false;
        break;
      }
    }
    if (equal) return offset;
  }
  return -1;
}

// Every light reaches every item, so the items differ only by channel mask.
// One slice is built per distinct mask and shared by the items carrying it,
// which collapses to a single slice at offset 0 for the usual all-on scene.
LightCullResult _assignInfiniteLights(
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

// Reused key buffers for [_keepNearest] (one slot per kept light).
Float64List _nearestDistance = Float64List(0);
Float64List _nearestCenter = Float64List(0);
Int32List _nearestIndex = Int32List(0);

// Moves the [keep] lights of [scratch] that reach [item] most strongly to the
// front of [scratch], nearest first. The order is lexicographic on (distance
// to the item's bounds, distance to the bounds' center, row index):
// directional lights first (they have no position), then the nearest local
// lights; lights inside or touching a large item's bounds all have zero
// distance, so the one nearest its center wins before row order decides.
// Row indices are unique, so this is a strict total order and the kept
// prefix equals that of fully sorting [scratch] with the same order, at
// O(n * keep) worst case instead of O(n log n) comparator calls that each
// recompute both distances (WR-81c).
void _keepNearest(
  RenderItem item,
  List<int> scratch,
  Map<int, CullableLight> lightsByIndex,
  int keep,
) {
  if (_nearestIndex.length < keep) {
    _nearestDistance = Float64List(keep);
    _nearestCenter = Float64List(keep);
    _nearestIndex = Int32List(keep);
  }
  final keyD = _nearestDistance;
  final keyC = _nearestCenter;
  final keyI = _nearestIndex;
  final bounds = item.worldBounds;
  var centerX = 0.0, centerY = 0.0, centerZ = 0.0;
  if (bounds != null) {
    centerX = (bounds.min.x + bounds.max.x) * 0.5;
    centerY = (bounds.min.y + bounds.max.y) * 0.5;
    centerZ = (bounds.min.z + bounds.max.z) * 0.5;
  }
  var held = 0;
  for (final index in scratch) {
    final position = lightsByIndex[index]!.worldPosition;
    final d = _distanceToItemSquared(item, bounds, position);
    // The center term only breaks distance ties between positioned lights on
    // a bounded item; equal distances otherwise mean two directionals (both
    // -infinity), whose center terms are both zero here.
    final c = bounds != null && position != null
        ? _distanceSquared(position, centerX, centerY, centerZ)
        : 0.0;
    var slot = held;
    if (held == keep) {
      if (_compareKeys(
            d,
            c,
            index,
            keyD[keep - 1],
            keyC[keep - 1],
            keyI[keep - 1],
          ) >=
          0) {
        continue;
      }
      slot = keep - 1;
    } else {
      held++;
    }
    while (slot > 0 &&
        _compareKeys(
              d,
              c,
              index,
              keyD[slot - 1],
              keyC[slot - 1],
              keyI[slot - 1],
            ) <
            0) {
      keyD[slot] = keyD[slot - 1];
      keyC[slot] = keyC[slot - 1];
      keyI[slot] = keyI[slot - 1];
      slot--;
    }
    keyD[slot] = d;
    keyC[slot] = c;
    keyI[slot] = index;
  }
  for (var i = 0; i < held; i++) {
    scratch[i] = keyI[i];
  }
}

int _compareKeys(double d1, double c1, int i1, double d2, double c2, int i2) {
  final byDistance = d1.compareTo(d2);
  if (byDistance != 0) return byDistance;
  final byCenter = c1.compareTo(c2);
  if (byCenter != 0) return byCenter;
  return i1.compareTo(i2);
}

/// Remembers the last [assignLightsToItems] result and hands it back while
/// its inputs are unchanged, so a frame whose lights and items did not change
/// (a moving camera included: the assignment is view-independent) costs one
/// comparison pass instead of a BVH scatter plus a nearest-light selection for
/// every overflowing item (WR-81c: ~500 lamps in a hotel spent 41 % of the
/// UI isolate's samples re-sorting identical lists every frame).
///
/// The inputs compared are everything the assignment reads: the [Bvh]
/// (identity and [Bvh.revision]), the item list (identity and order) with
/// each item's [RenderItem.worldBounds] (or, unbounded, its transform's
/// translation), [RenderItem.frustumCulled] and
/// [RenderItem.lightChannelMask], and every light's row index, influence
/// bounds, position, and channel mask, plus `maxPerItem`. A hit leaves the
/// items' [RenderItem.lightListOffset]/[RenderItem.lightListCount] as the
/// last assignment wrote them; only the culler writes those fields.
class LightAssignmentCache {
  /// How many times [assign] actually ran the assignment.
  int get recomputeCount => _recomputeCount;
  int _recomputeCount = 0;

  LightCullResult? _result;
  Bvh? _bvh;
  int _bvhRevision = -1;
  int _maxPerItem = -1;
  List<RenderItem> _items = const [];
  Float64List _itemKeys = Float64List(0);
  Float64List _lightKeys = Float64List(0);

  static const int _floatsPerItem = 8;
  static const int _floatsPerLight = 13;

  /// [assignLightsToItems], skipped when nothing it reads changed since the
  /// previous call.
  LightCullResult assign({
    required List<RenderItem> items,
    required Bvh bvh,
    required List<CullableLight> lights,
    required int maxPerItem,
  }) {
    final cached = _result;
    if (cached != null &&
        identical(bvh, _bvh) &&
        bvh.revision == _bvhRevision &&
        maxPerItem == _maxPerItem &&
        _sameLights(lights) &&
        _sameItems(items)) {
      return cached;
    }
    _recomputeCount++;
    final result = assignLightsToItems(
      items: items,
      bvh: bvh,
      lights: lights,
      maxPerItem: maxPerItem,
    );
    _result = result;
    _bvh = bvh;
    _bvhRevision = bvh.revision;
    _maxPerItem = maxPerItem;
    _items = List<RenderItem>.of(items, growable: false);
    _itemKeys = Float64List(items.length * _floatsPerItem);
    for (var i = 0; i < items.length; i++) {
      _writeItemKey(items[i], _itemKeys, i * _floatsPerItem);
    }
    _lightKeys = Float64List(lights.length * _floatsPerLight);
    for (var i = 0; i < lights.length; i++) {
      _writeLightKey(lights[i], _lightKeys, i * _floatsPerLight);
    }
    return result;
  }

  /// Forgets the remembered assignment, so the next [assign] recomputes.
  void clear() {
    _result = null;
    _bvh = null;
    _items = const [];
  }

  // Scratch for comparing one key without allocating.
  final Float64List _probe = Float64List(_floatsPerLight);

  bool _sameItems(List<RenderItem> items) {
    if (items.length != _items.length) return false;
    final keys = _itemKeys;
    final probe = _probe;
    for (var i = 0; i < items.length; i++) {
      final item = items[i];
      if (!identical(item, _items[i])) return false;
      _writeItemKey(item, probe, 0);
      final base = i * _floatsPerItem;
      for (var k = 0; k < _floatsPerItem; k++) {
        // NaN never equals itself, so a NaN bound always recomputes.
        if (probe[k] != keys[base + k]) return false;
      }
    }
    return true;
  }

  bool _sameLights(List<CullableLight> lights) {
    if (lights.length * _floatsPerLight != _lightKeys.length) return false;
    final keys = _lightKeys;
    final probe = _probe;
    for (var i = 0; i < lights.length; i++) {
      _writeLightKey(lights[i], probe, 0);
      final base = i * _floatsPerLight;
      for (var k = 0; k < _floatsPerLight; k++) {
        if (probe[k] != keys[base + k]) return false;
      }
    }
    return true;
  }

  static void _writeItemKey(RenderItem item, Float64List out, int o) {
    final bounds = item.worldBounds;
    out[o] = (bounds != null ? 1.0 : 0.0) + (item.frustumCulled ? 2.0 : 0.0);
    out[o + 1] = item.lightChannelMask.toDouble();
    if (bounds != null) {
      out[o + 2] = bounds.min.x;
      out[o + 3] = bounds.min.y;
      out[o + 4] = bounds.min.z;
      out[o + 5] = bounds.max.x;
      out[o + 6] = bounds.max.y;
      out[o + 7] = bounds.max.z;
    } else {
      final t = item.worldTransform.storage;
      out[o + 2] = t[12];
      out[o + 3] = t[13];
      out[o + 4] = t[14];
      out[o + 5] = 0.0;
      out[o + 6] = 0.0;
      out[o + 7] = 0.0;
    }
  }

  static void _writeLightKey(CullableLight light, Float64List out, int o) {
    final bounds = light.bounds;
    final position = light.worldPosition;
    out[o] = light.index.toDouble();
    out[o + 1] = light.channelMask.toDouble();
    out[o + 2] = (bounds != null ? 1.0 : 0.0) + (position != null ? 2.0 : 0.0);
    out[o + 3] = bounds?.min.x ?? 0.0;
    out[o + 4] = bounds?.min.y ?? 0.0;
    out[o + 5] = bounds?.min.z ?? 0.0;
    out[o + 6] = bounds?.max.x ?? 0.0;
    out[o + 7] = bounds?.max.y ?? 0.0;
    out[o + 8] = bounds?.max.z ?? 0.0;
    out[o + 9] = position?.x ?? 0.0;
    out[o + 10] = position?.y ?? 0.0;
    out[o + 11] = position?.z ?? 0.0;
    out[o + 12] = 0.0;
  }
}

double _distanceToItemSquared(
  RenderItem item,
  Aabb3? bounds,
  Vector3? position,
) {
  // Directional lights have no position and must survive any local-light cap.
  if (position == null) return double.negativeInfinity;
  if (bounds == null) {
    final transform = item.worldTransform.storage;
    return _distanceSquared(
      position,
      transform[12],
      transform[13],
      transform[14],
    );
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

double _distanceSquared(Vector3 position, double x, double y, double z) {
  final dx = position.x - x;
  final dy = position.y - y;
  final dz = position.z - z;
  return dx * dx + dy * dy + dz * dz;
}
