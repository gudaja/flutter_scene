// Loading a shader library issues compileShader without asking for the compile
// status (a synchronous GPU round trip per shader); a shader that failed to
// compile is reported, with its info log and source, by the link of the first
// pipeline that uses it. A hot reload still checks at once, so a broken edit
// never replaces a working shader.
//
//   flutter test --platform chrome \
//     test/gpu/webgl/webgl_deferred_compile_browser_test.dart
//
// ignore_for_file: implementation_imports
@TestOn('browser')
library;

import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:flutter_scene/src/gpu/web/_gpu.dart' as gpu;
import 'package:flutter_scene/src/gpu/web/shader_bundle_generated.dart' as fb;
import 'package:flutter_scene/src/gpu/web/webgl/_webgl.dart';
import 'package:flutter_test/flutter_test.dart';

const String _vertex =
    '#version 100\nattribute vec2 position;\n'
    'void main() { gl_Position = vec4(position, 0.0, 1.0); }\n';
const String _white =
    '#version 100\nprecision mediump float;\n'
    'void main() { gl_FragColor = vec4(1.0); }\n';
// Uses an identifier nobody declared: fails to compile.
const String _broken =
    '#version 100\nprecision mediump float;\n'
    'void main() { gl_FragColor = vec4(undeclaredWr81h6); }\n';

Uint8List _bundle({required String brokenSource}) {
  fb.ShaderObjectBuilder shader(
    String name,
    fb.ShaderStage stage,
    String source, {
    List<fb.ShaderInputObjectBuilder> inputs = const [],
  }) => fb.ShaderObjectBuilder(
    name: name,
    openglEs: fb.BackendShaderObjectBuilder(
      stage: stage,
      entrypoint: 'main',
      inputs: inputs,
      uniformStructs: const [],
      uniformTextures: const [],
      shader: utf8.encode(source),
    ),
  );
  return fb.ShaderBundleObjectBuilder(
    formatVersion: 2,
    shaders: [
      shader(
        'PointVertex',
        fb.ShaderStage.kVertex,
        _vertex,
        inputs: [fb.ShaderInputObjectBuilder(name: 'position')],
      ),
      shader('WhiteFragment', fb.ShaderStage.kFragment, _white),
      shader('BrokenFragment', fb.ShaderStage.kFragment, brokenSource),
    ],
  ).toBytes();
}

const gpu.VertexLayout _layout = gpu.VertexLayout(
  buffers: [
    gpu.VertexBuffer(
      strideInBytes: 8,
      attributes: [
        gpu.VertexAttribute(
          name: 'position',
          format: gpu.VertexFormat.float32x2,
        ),
      ],
    ),
  ],
);

/// Polls [pipeline] until it is ready; rethrows its link failure.
Future<bool> _whenReady(gpu.RenderPipeline pipeline) async {
  for (var i = 0; i < 500; i++) {
    if (gpu.gpuHost.isPipelineReady(pipeline)) return true;
    await Future<void>.delayed(const Duration(milliseconds: 5));
  }
  return false;
}

void main() {
  if (gpu.useWebGpuBackend) {
    test('needs the WebGL2 shim', () {
      markTestSkipped('The shim is WebGPU in this build.');
    });
    return;
  }

  // Counts gl.getShaderParameter calls (COMPILE_STATUS is the only one the
  // backend asks for).
  var statusQueries = 0;
  late gpu.ShaderLibrary library;
  setUpAll(() async {
    await gpu.initializeGpuBackend();
    final gl = (gpu.gpuContext as WebGlContext).gl;
    final proto = (gl as JSObject).getProperty<JSObject>('__proto__'.toJS);
    final original = proto.getProperty<JSFunction>('getShaderParameter'.toJS);
    (gl as JSObject).setProperty(
      'getShaderParameter'.toJS,
      ((JSObject shader, JSNumber pname) {
        statusQueries++;
        return original.callAsFunction(gl, shader, pname);
      }).toJS,
    );
  });
  tearDownAll(() {
    final gl = (gpu.gpuContext as WebGlContext).gl;
    (gl as JSObject).delete('getShaderParameter'.toJS);
  });

  test('T-WR81h6-4 · loading a library (even with a broken shader) asks '
      'for no compile status', () async {
    library = (await gpu.loadShaderLibraryFromBytesAsync(
      ByteData.sublistView(_bundle(brokenSource: _broken)),
    ))!;
    expect(library['BrokenFragment'], isNotNull);
    expect(statusQueries, 0);
  });

  test('T-WR81h6-5 · a pipeline of compiled shaders links and binds without '
      'a compile-status query', () async {
    final pipeline = gpu.gpuContext.createRenderPipeline(
      library['PointVertex']!,
      library['WhiteFragment']!,
      vertexLayout: _layout,
    );
    expect(await _whenReady(pipeline), isTrue);
    expect(statusQueries, 0);
  });

  test('T-WR81h6-6 · the link of a pipeline with a broken shader reports '
      'the compile failure with its info log and source', () async {
    Object? failure;
    try {
      final pipeline = gpu.gpuContext.createRenderPipeline(
        library['PointVertex']!,
        library['BrokenFragment']!,
        vertexLayout: _layout,
      );
      await _whenReady(pipeline);
    } on Exception catch (e) {
      failure = e;
    }
    expect('$failure', contains('Failed to link program'));
    expect('$failure', contains('Failed to compile fragment shader:'));
    expect('$failure', contains('undeclaredWr81h6'));
    expect('$failure', contains('undeclared identifier'));
    expect('$failure', contains('--- source ---'));
    expect('$failure', isNot(contains('Failed to compile vertex shader')));
  });

  test('T-WR81h6-7 · a hot reload with a broken shader fails at once and '
      'keeps the working shader', () async {
    final before = statusQueries;
    final good = ByteData.sublistView(_bundle(brokenSource: _white));
    expect(
      await gpu.reinitializeShaderLibraryFromBytesAsync(library, good),
      isNull,
    );
    expect(statusQueries, greaterThan(before));
    final white = library['WhiteFragment']! as WebGlShader;
    final glShader = white.glShader;
    // Swap WhiteFragment's source for the broken one.
    final bytes = fb.ShaderBundleObjectBuilder(
      formatVersion: 2,
      shaders: [
        fb.ShaderObjectBuilder(
          name: 'WhiteFragment',
          openglEs: fb.BackendShaderObjectBuilder(
            stage: fb.ShaderStage.kFragment,
            entrypoint: 'main',
            inputs: const [],
            uniformStructs: const [],
            uniformTextures: const [],
            shader: utf8.encode(_broken),
          ),
        ),
      ],
    ).toBytes();
    final error = await gpu.reinitializeShaderLibraryFromBytesAsync(
      library,
      ByteData.sublistView(bytes),
    );
    expect(error, contains('Failed to compile fragment shader'));
    expect(white.glShader, same(glShader));
  });
}
