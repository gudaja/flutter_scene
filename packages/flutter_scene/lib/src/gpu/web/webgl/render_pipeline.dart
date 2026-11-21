part of '_webgl.dart';

/// A linked WebGL program built from a vertex + fragment shader pair, plus
/// cached reflection state used by RenderPass when binding resources.
final class WebGlRenderPipeline extends RenderPipeline {
  WebGlRenderPipeline._(
    this._gpuContext,
    this.vertexShader,
    this.fragmentShader, {
    VertexLayout? vertexLayout,
  }) : _vertexLayout = vertexLayout {
    final gl = _gpuContext._gl;
    final program = gl.createProgram();
    if (program == null) {
      // Native flutter_gpu reports a pipeline it cannot build as an
      // Exception, and the renderer's guard is scoped to that, so a failure
      // here must not be an Error or it blanks the frame on WebGL2 alone.
      throw Exception('Failed to create WebGL program');
    }
    _program = program;

    final vs = vertexShader._glShader;
    final fs = fragmentShader._glShader;
    if (vs == null || fs == null) {
      throw Exception('Shader objects were not compiled before linking');
    }
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);

    // Bind attribute locations from the vertex shader's reflected inputs so
    // the layout is deterministic. Inline pipelines have no reflection data;
    // there the linker assigns locations and bindVertexBuffer looks them up
    // with getAttribLocation.
    for (final input in vertexShader.vertexInputs) {
      gl.bindAttribLocation(program, input.location, input.name);
    }

    // The shaders' compile status is read only if the link fails (see
    // WebGlShader._compile); captured now, as a hot reload may replace them.
    final compiles = [
      ?vertexShader._compileState,
      ?fragmentShader._compileState,
    ];
    gl.linkProgram(program);
    // Reading LINK_STATUS right after linkProgram waits for the whole link
    // (and every command queued before it) on the calling thread. With
    // KHR_parallel_shader_compile the link runs in the background instead:
    // the pipeline stays pending until a non-blocking COMPLETION_STATUS_KHR
    // query reports it done (see isLinkReady), and only then are the status
    // and the reflection read. Without the extension this links at once.
    _link = ProgramLink(
      parallel: _gpuContext._parallelShaderCompile,
      isComplete: () =>
          (gl.getProgramParameter(program, _kCompletionStatusKhr) as JSBoolean?)
              ?.toDart ??
          true,
      linkError: () {
        final ok =
            (gl.getProgramParameter(
                      program,
                      web.WebGL2RenderingContext.LINK_STATUS,
                    )
                    as JSBoolean?)
                ?.toDart ??
            false;
        if (ok) {
          for (final compile in compiles) {
            compile.markCompiled();
          }
          return null;
        }
        // A shader that failed to compile fails the link: name it, with its
        // info log, as a compile check right after compileShader would have.
        final log = linkFailureLog(gl.getProgramInfoLog(program), compiles);
        gl.deleteProgram(program);
        _gpuContext._forgetPipeline(this);
        return log;
      },
      onLinked: _buildUniformCaches,
      nowMicros: _gpuContext._clockMicros,
    );
  }

  /// `COMPLETION_STATUS_KHR` (KHR_parallel_shader_compile).
  static const int _kCompletionStatusKhr = 0x91B1;

  late final ProgramLink _link;

  /// Whether the program has linked, without waiting for a link that still
  /// runs in the background. Throws the link failure.
  bool get isLinkReady => _link.poll();

  /// Completes a background link now, waiting for it (the path of a draw that
  /// binds a pipeline nobody polled). Throws the link failure.
  void _finishLink() => _link.finish();

  final WebGlContext _gpuContext;
  @override
  final WebGlShader vertexShader;
  @override
  final WebGlShader fragmentShader;
  // ignore: unused_field
  final VertexLayout? _vertexLayout;

  /// The explicit vertex layout this pipeline was created with, or null for
  /// the default reflection-derived single-buffer layout.
  VertexLayout? get vertexLayout => _vertexLayout;
  late final web.WebGLProgram _program;

  /// GL uniform locations keyed by "structName.memberName".
  final Map<String, web.WebGLUniformLocation?> _memberLocations = {};

  /// Member locations per struct type name, parallel to the struct's member
  /// list, so the per-draw uniform bind avoids building lookup strings.
  final Map<String, List<web.WebGLUniformLocation?>> _structLocations = {};

  /// Uniform-block binding point per struct type name. Populated when the
  /// linked program exposes the struct as a real uniform block (GLSL ES
  /// 3.00 bundles), in which case the per-draw bind uses bindBufferRange
  /// instead of per-member glUniform calls.
  final Map<String, int> _structBlockBindings = {};

  /// Minimum bind size per block, the larger of the reflected std140 size and
  /// the linker-reported UNIFORM_BLOCK_DATA_SIZE. Drivers may pad the block
  /// beyond the reflected size (Mesa rounds up to a vec4 multiple); binding a
  /// smaller range than the driver reports makes WebGL2 reject the draw.
  final Map<String, int> _structBlockBindSizes = {};

  /// The largest block bind size, used to size the zero fallback buffer bound
  /// to every block binding point.
  int _maxBlockSizeBytes = 0;

  /// A zero-filled uniform buffer bound to every active block binding point
  /// when this pipeline is bound (see [_bindDefaultUniformBlocks]). Created
  /// lazily on first pipeline bind.
  web.WebGLBuffer? _defaultBlockBuffer;

  /// Texture unit assigned to each reflected sampler name.
  final Map<String, int> _samplerUnits = {};

  web.WebGLProgram get glProgram => _program;

  void _buildUniformCaches() {
    final gl = _gpuContext._gl;
    gl.useProgram(_program);
    try {
      _buildUniformCachesOnCurrentProgram(gl);
    } finally {
      // A background link can finish mid-pass, between another pipeline's
      // bind and its draws, so give that pipeline its program back.
      final bound = _gpuContext._boundProgram;
      if (bound != null) gl.useProgram(bound);
    }
  }

  void _buildUniformCachesOnCurrentProgram(web.WebGL2RenderingContext gl) {
    final yFlipLocation = gl.getUniformLocation(_program, '_impeller_y_flip');
    if (yFlipLocation != null) {
      // Impeller's generated GLES shaders multiply gl_Position.y by this
      // backend uniform. The web shim does not use Impeller's runtime uniform
      // binding layer, so set the value that stores FBO render targets
      // top-down, matching Flutter GPU's render-to-texture convention.
      gl.uniform1f(yFlipLocation, -1.0);
    }
    var nextUnit = 0;
    var nextBlockBinding = 0;
    for (final shader in [vertexShader, fragmentShader]) {
      for (final struct in shader.uniformStructs) {
        if (_structBlockBindings.containsKey(struct.name) ||
            _structLocations.containsKey(struct.name)) {
          continue; // already resolved via the other stage
        }
        // GLSL ES 3.00 bundles emit uniform structs as std140 uniform
        // blocks (named by type). Assign each block a binding point once;
        // the per-draw bind attaches the data with bindBufferRange.
        final blockIndex = gl.getUniformBlockIndex(_program, struct.name);
        if (blockIndex != web.WebGL2RenderingContext.INVALID_INDEX) {
          final binding = nextBlockBinding++;
          gl.uniformBlockBinding(_program, blockIndex, binding);
          _structBlockBindings[struct.name] = binding;
          final dataSize =
              (gl.getActiveUniformBlockParameter(
                        _program,
                        blockIndex,
                        web.WebGL2RenderingContext.UNIFORM_BLOCK_DATA_SIZE,
                      )
                      as JSNumber?)
                  ?.toDartInt ??
              struct.sizeInBytes;
          final bindSize = dataSize > struct.sizeInBytes
              ? dataSize
              : struct.sizeInBytes;
          _structBlockBindSizes[struct.name] = bindSize;
          if (bindSize > _maxBlockSizeBytes) {
            _maxBlockSizeBytes = bindSize;
          }
          continue;
        }
        // GLSL ES 1.00 bundles flatten the struct to plain uniforms. GL
        // uniform names use the struct's instance name; reflection and
        // callers use its type name. Look up via instance, key by type.
        final instance =
            shader._structInstanceNames[struct.name] ?? struct.name;
        final locations = <web.WebGLUniformLocation?>[];
        for (final m in struct.members) {
          final glName = '$instance.${m.name}';
          final loc =
              gl.getUniformLocation(_program, glName) ??
              gl.getUniformLocation(_program, '$glName[0]');
          _memberLocations['${struct.name}.${m.name}'] = loc;
          locations.add(loc);
        }
        _structLocations[struct.name] = locations;
      }
      for (final tex in shader.textureBindings) {
        if (_samplerUnits.containsKey(tex.name)) continue;
        final unit = nextUnit++;
        _samplerUnits[tex.name] = unit;
        final loc = gl.getUniformLocation(_program, tex.name);
        if (loc != null) {
          // Bind the sampler uniform to its texture unit once at link time.
          gl.uniform1i(loc, unit);
        }
      }
    }
  }

  /// Binds a zero-filled fallback buffer to every active uniform block binding
  /// point. Called when this pipeline is bound, before the caller binds its
  /// own uniforms.
  ///
  /// A block that the linker keeps active but the caller never binds (for
  /// example the framework's `RadianceLayoutInfo`, which an unlit custom
  /// material's shader includes but never binds) is a used-but-unbound uniform
  /// buffer. Impeller tolerates that; WebGL2 raises an error and drops the
  /// draw. Pre-binding a zero buffer makes the unbound block read zeros
  /// instead. Explicit [RenderPass.bindUniform] calls override the binding
  /// points they set.
  void _bindDefaultUniformBlocks() {
    if (_structBlockBindings.isEmpty) return;
    final gl = _gpuContext._gl;
    var buffer = _defaultBlockBuffer;
    if (buffer == null) {
      // std140 blocks round up to 16 bytes; keep a floor so the buffer is
      // never zero-sized, and size it to the largest block so the range is
      // always big enough for any binding point it is attached to.
      final size = _maxBlockSizeBytes < 16
          ? 16
          : (_maxBlockSizeBytes + 15) & ~15;
      buffer = gl.createBuffer();
      if (buffer == null) {
        throw StateError('Failed to create WebGL buffer');
      }
      gl.bindBuffer(web.WebGL2RenderingContext.UNIFORM_BUFFER, buffer);
      gl.bufferData(
        web.WebGL2RenderingContext.UNIFORM_BUFFER,
        Uint8List(size).toJS,
        web.WebGL2RenderingContext.STATIC_DRAW,
      );
      _defaultBlockBuffer = buffer;
    }
    for (final binding in _structBlockBindings.values) {
      gl.bindBufferBase(
        web.WebGL2RenderingContext.UNIFORM_BUFFER,
        binding,
        buffer,
      );
    }
  }
}
