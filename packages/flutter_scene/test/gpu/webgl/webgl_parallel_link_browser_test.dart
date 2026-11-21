// A program linked under KHR_parallel_shader_compile becomes ready without the
// caller waiting on the link, a failed link still reports its log, and a
// render target recreated with the same formats skips the completeness check.
//
//   flutter test --platform chrome \
//     test/gpu/webgl/webgl_parallel_link_browser_test.dart
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
import 'package:web/web.dart' as web;

Uint8List _bundle() {
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
        '#version 100\nattribute vec2 position;\n'
            'void main() { gl_Position = vec4(position, 0.0, 1.0); }\n',
        inputs: [fb.ShaderInputObjectBuilder(name: 'position')],
      ),
      shader(
        'WhiteFragment',
        fb.ShaderStage.kFragment,
        '#version 100\nprecision mediump float;\n'
            'void main() { gl_FragColor = vec4(1.0); }\n',
      ),
      // Reads a varying the vertex stage never writes: compiles, fails to
      // link.
      shader(
        'VaryingFragment',
        fb.ShaderStage.kFragment,
        '#version 100\nprecision mediump float;\nvarying vec4 v_missing;\n'
            'void main() { gl_FragColor = v_missing; }\n',
      ),
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

void main() {
  if (gpu.useWebGpuBackend) {
    test('needs the WebGL2 shim', () {
      markTestSkipped('The shim is WebGPU in this build.');
    });
    return;
  }

  late Map<String, gpu.Shader> library;
  setUpAll(() async {
    await gpu.initializeGpuBackend();
    final lib = (await gpu.loadShaderLibraryFromBytesAsync(
      ByteData.sublistView(_bundle()),
    ))!;
    library = {
      for (final name in ['PointVertex', 'WhiteFragment', 'VaryingFragment'])
        name: lib[name]!,
    };
  });

  test('T-WR81h1-7 · a pipeline becomes ready by polling, and binds', () async {
    final pipeline = gpu.gpuContext.createRenderPipeline(
      library['PointVertex']!,
      library['WhiteFragment']!,
      vertexLayout: _layout,
    );
    var ready = false;
    for (var i = 0; i < 500 && !ready; i++) {
      ready = gpu.gpuHost.isPipelineReady(pipeline);
      if (!ready) await Future<void>.delayed(const Duration(milliseconds: 5));
    }
    expect(ready, isTrue);
    final color = gpu.gpuContext.createTexture(
      gpu.StorageMode.devicePrivate,
      4,
      4,
    );
    final commands = gpu.gpuContext.createCommandBuffer();
    final pass = commands.createRenderPass(
      gpu.RenderTarget.singleColor(gpu.ColorAttachment(texture: color)),
    );
    pass.bindPipeline(pipeline);
    final gl = (gpu.gpuContext as WebGlContext).gl;
    expect(gl.getError(), web.WebGL2RenderingContext.NO_ERROR);
    commands.submit();
  });

  test('T-WR81h1-8 · a failed link is reported with its log', () async {
    Object? failure;
    try {
      final pipeline = gpu.gpuContext.createRenderPipeline(
        library['PointVertex']!,
        library['VaryingFragment']!,
        vertexLayout: _layout,
      );
      for (var i = 0; i < 500; i++) {
        if (gpu.gpuHost.isPipelineReady(pipeline)) break;
        await Future<void>.delayed(const Duration(milliseconds: 5));
      }
    } on Exception catch (e) {
      failure = e;
    }
    expect('$failure', contains('Failed to link program'));
  });

  test('T-WR81h1-9 · a new render target with known formats skips '
      'checkFramebufferStatus', () {
    final gl = (gpu.gpuContext as WebGlContext).gl;
    final proto = (gl as JSObject).getProperty<JSObject>('__proto__'.toJS);
    final original = proto.getProperty<JSFunction>(
      'checkFramebufferStatus'.toJS,
    );
    var checks = 0;
    (gl as JSObject).setProperty(
      'checkFramebufferStatus'.toJS,
      ((JSNumber target) {
        checks++;
        return original.callAsFunction(gl, target);
      }).toJS,
    );
    try {
      void pass(int size) {
        final color = gpu.gpuContext.createTexture(
          gpu.StorageMode.devicePrivate,
          size,
          size,
        );
        final commands = gpu.gpuContext.createCommandBuffer();
        commands.createRenderPass(
          gpu.RenderTarget.singleColor(gpu.ColorAttachment(texture: color)),
        );
        commands.submit();
      }

      pass(8);
      final first = checks;
      pass(16);
      pass(32);
      expect(first, lessThanOrEqualTo(1));
      expect(checks, first);
    } finally {
      (gl as JSObject).delete('checkFramebufferStatus'.toJS);
    }
  });
}
