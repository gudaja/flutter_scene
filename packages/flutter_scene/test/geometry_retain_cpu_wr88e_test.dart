// WR-88e (home_3d_viwer): geometry without a retained CPU copy.
//
// The app uploads its plan meshes with `retainCpuData: false`: nothing reads
// them back (picking and cutaway run on the app's own mesh data), and on an
// estate of 24 houses the retained structure-of-arrays copies (72 bytes per
// vertex per geometry) were most of the Dart heap. These tests pin both sides
// of the option: the default keeps independent, readable copies; `false`
// keeps none, yet bounds and vertex count match. GPU-gated: upload needs a
// device.

import 'dart:typed_data';

import 'package:flutter_scene/scene.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/gpu_available.dart';

MeshData _quad() => MeshData.build(
  positions: Float32List.fromList([0, 0, 0, 2, 0, 0, 2, 3, 0, 0, 3, -1]),
  texCoords: Float32List.fromList([0, 0, 1, 0, 1, 1, 0, 1]),
  indices: [0, 1, 2, 0, 2, 3],
);

void main() {
  // `Scene()` (the GPU check) needs the binding.
  TestWidgetsFlutterBinding.ensureInitialized();
  bool noGpu() {
    if (gpuAvailable()) return false;
    markTestSkipped('needs a GPU device');
    return true;
  }

  test('T-WR88e-F1 · default keeps an independent, readable CPU copy', () {
    if (noGpu()) return;
    final data = _quad();
    final geometry = MeshGeometry.fromMeshData(data);
    expect(geometry.isReadable, isTrue);
    data.positions[0] = 99; // the caller's array is not the retained one
    final back = geometry.extractMeshData();
    expect(back.positions[0], 0);
    expect(back.vertexCount, 4);
    expect(back.indices, [0, 1, 2, 0, 2, 3]);
    expect(geometry.cpuMeshData.positions, isNotNull);
  });

  test('T-WR88e-F2 · retainCpuData: false keeps no CPU data, same bounds '
      'and vertex count', () {
    if (noGpu()) return;
    final kept = MeshGeometry.fromMeshData(_quad());
    final bare = MeshGeometry.fromMeshData(_quad(), retainCpuData: false);
    expect(bare.retainCpuData, isFalse);
    expect(bare.isReadable, isFalse);
    final cpu = bare.cpuMeshData;
    expect(cpu.positions, isNull);
    expect(cpu.vertices, isNull);
    expect(cpu.indices, isNull);
    expect(bare.extractMeshData, throwsStateError);
    final node = Node()..mesh = Mesh(bare, UnlitMaterial());
    expect(node.extractMeshData, throwsStateError);
    expect(bare.vertexCount, kept.vertexCount);
    expect(bare.localBounds!.min, kept.localBounds!.min);
    expect(bare.localBounds!.max, kept.localBounds!.max);
    expect(bare.localBoundingSphere!.radius, kept.localBoundingSphere!.radius);
    expect(bare.vertexStreamCount, kept.vertexStreamCount);
  });
}
