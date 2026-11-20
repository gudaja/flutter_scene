import 'dart:typed_data';

import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:vector_math/vector_math.dart';

import 'package:flutter_scene/src/gpu/render_pass_compat.dart';
import 'package:flutter_scene/src/light.dart';
import 'package:flutter_scene/src/material/material.dart' show Material;
import 'package:flutter_scene/src/render/point_shadow.dart';
import 'package:flutter_scene/src/render/render_graph.dart';
import 'package:flutter_scene/src/render/render_scene.dart';
import 'package:flutter_scene/src/render/shadow_cache.dart';
import 'package:flutter_scene/src/render/shadow_encoder.dart';
import 'package:flutter_scene/src/render/frame_transients.dart';
import 'package:flutter_scene/src/render/spot_shadow.dart';
import 'package:flutter_scene/src/shaders.dart';
import 'package:flutter_scene/src/scene_encoder.dart'
    show deferredPipelineBuilds, resolvePipeline;
import 'package:flutter_scene/src/render/uniform_slots.dart';

/// Render-graph blackboard key under which [ShadowPass] publishes the shadow
/// map atlas (a depth-in-`.r` fp32 texture). The downstream scene pass reads it
/// from here.
const String kShadowMapBlackboardKey = 'directional_shadow_map';

/// Blackboard key for the frame's [SpotShadowInfo], set when spots cast.
const String kSpotShadowInfoBlackboardKey = 'spot_shadow_info';

/// Where the frame's spot shadows live in the shared shadow atlas (the
/// texture under [kShadowMapBlackboardKey]): one square tile per casting spot,
/// after the directional cascades, holding window-space depth in red.
/// {@category Rendering}
class SpotShadowInfo {
  /// Creates the layout record for one frame's spot shadows.
  const SpotShadowInfo({
    required this.matrices,
    required this.firstTile,
    required this.totalTiles,
    required this.tileResolution,
    int? lampTileResolution,
  }) : lampTileResolution = lampTileResolution ?? tileResolution;

  /// World to clip matrix of each casting spot, in shadow slot order.
  final List<Matrix4> matrices;

  /// Atlas tile index of slot 0; slot `i` is tile `firstTile + i`.
  final int firstTile;

  /// Tiles across the atlas (its width is `totalTiles * tileResolution`).
  final int totalTiles;

  /// Edge length of a tile in texels.
  final int tileResolution;

  /// Edge length of a spot or point tile in texels. Smaller than
  /// [tileResolution] when the lights render at their own resolution
  /// (`Scene.punctualShadowsAtOwnResolution`); such tiles stack
  /// `tileResolution ~/ lampTileResolution` to a column after the cascades,
  /// and [totalTiles] still counts the cascades plus the spot and point tiles.
  final int lampTileResolution;
}

/// The spot and point shadow tile edge to use beside cascade tiles of
/// [cascadeResolution] when the lights ask for [ownResolution]: the light's
/// own when it is smaller and divides the cascade edge by a power of two (so
/// the tiles stack in whole columns), else the cascade edge.
int punctualShadowTileResolution(int cascadeResolution, int ownResolution) {
  if (ownResolution <= 0 || ownResolution >= cascadeResolution) {
    return cascadeResolution;
  }
  if (cascadeResolution % ownResolution != 0) return cascadeResolution;
  final ratio = cascadeResolution ~/ ownResolution;
  return ratio & (ratio - 1) == 0 ? ownResolution : cascadeResolution;
}

/// Forces a site shadow map (WR-77) through the per-frame atlas composite
/// instead of publishing its cached tile, so parity tests can render both.
bool debugDisableSiteShadowTilePublish = false;

/// Diagnostic (WR-74): how the lit shaders sample the directional shadow.
/// 0 = normal; 1 = skip the lookup (shadow variant still bound, reads lit);
/// 2 = the five probe taps only (no penumbra kernel); 3 = one center tap.
/// Modes 1-3 change the image and exist only to split the per-pixel cost.
int debugShadowSamplingMode = 0;

/// Allocates and renders the shadow atlas even when no caster would draw into
/// it, so parity tests can render both against the placeholder (WR-88b).
bool debugAlwaysAllocateShadowAtlas = false;

/// Forces every view to render its own spot and point shadow tiles, so
/// parity tests can render the same frame with and without sharing.
bool debugDisableSharedShadowTiles = false;

/// One frame's spot and point shadow tiles, shared by its texture views and
/// its screen view.
///
/// The casters are drawn once, from [cameraPosition] (the screen view's
/// camera), so a mirror or render texture shows the shadows the screen view
/// sees. The first view to render its [ShadowPass] fills this and later
/// views publish the same atlas. Only frames without directional cascades
/// share, since cascades fit each view.
// TODO(shared-shadow-opt-out): let a texture view that needs shadows drawn
// from its own camera (a distant monitor view, say) render its own tiles.
class SharedShadowAtlas {
  /// Creates an empty atlas whose casters are drawn from [cameraPosition].
  SharedShadowAtlas(this.cameraPosition);

  /// The camera position every sharing view's casters are drawn from.
  final Vector3 cameraPosition;

  /// The atlas the first view rendered, or null before any view has.
  gpu.Texture? atlas;

  /// The spot layout published with [atlas].
  SpotShadowInfo? spotInfo;
}

/// Render-graph blackboard key under which [ShadowPass] publishes the packed
/// shadow uniform (the `PostShadowInfo` std140 block: per-cascade world->light
/// matrices, split distances, the light direction + cascade count, and the
/// light color). A depth-aware custom pass reads it to sample the shadow map.
const String kShadowUniformBlackboardKey = 'shadow_uniform';

/// Renders the scene's depth into one shared shadow map atlas and publishes it
/// on the render-graph blackboard: the directional light's cascades first, then
/// each shadow-casting spot's cone, then each shadow-casting point light's six
/// cube faces (quarter-tile quadrants, two tiles per light).
///
/// The atlas is one fp32 color texture holding the tiles as a horizontal strip,
/// each [tileResolution] square (cascade tiles `0..cascades.length`, then spot
/// tiles, then point tiles); window-space depth goes in the red channel (a
/// transient depth attachment backs the depth test). It is cleared to 1.0 so
/// texels no caster covers read as "lit". Sharing one atlas keeps every shadow
/// type on a single sampler in the lit shader.
///
/// With a [cachePlan] (some casters are `shadowStatic`), static casters render
/// into the plan's persistent per-cascade tiles only when the plan asks, and
/// each frame's cascade tile is composed by replaying the cached tile (color
/// and fragment depth) and drawing just the dynamic casters on top. Without a
/// plan every caster renders every frame.
class ShadowPass extends RenderGraphPass {
  ShadowPass({
    required RenderScene renderScene,
    required int tileResolution,
    required Vector3 cameraPosition,
    List<ShadowCascade> cascades = const [],
    List<List<Plane>> cascadeReceiverPlanes = const [],
    ShadowCasterFaces casterFaces = ShadowCasterFaces.front,
    int casterChannelMask = 0xFF,
    SpotShadowFrame? spotShadows,
    PointShadowFrame? pointShadows,
    ByteData? shadowUniform,
    ShadowCachePlan? cachePlan,
    SharedShadowAtlas? shared,
    bool siteShadow = false,
    double siteMinCasterSize = 0.0,
    int? punctualTileResolution,
  }) : assert(shared == null || cascades.isEmpty),
       _punctualTileResolution = punctualTileResolution ?? tileResolution,
       _siteShadow = siteShadow,
       _siteMinCasterSize = siteShadow ? siteMinCasterSize : 0.0,
       _renderScene = renderScene,
       _cascades = cascades,
       _cascadeReceiverPlanes = cascadeReceiverPlanes,
       _tileResolution = tileResolution,
       _casterFaces = casterFaces,
       _casterChannelMask = casterChannelMask,
       _cameraPosition = cameraPosition,
       _spotShadows = spotShadows,
       _pointShadows = pointShadows,
       _shadowUniform = shadowUniform,
       _cachePlan = cachePlan,
       _shared = shared;

  final RenderScene _renderScene;
  final List<ShadowCascade> _cascades;

  // WR-77: the cascade is a site shadow map, whose casters write a
  // slope-biased depth (spot and point tiles keep the plain depth).
  final bool _siteShadow;
  final double _siteMinCasterSize;

  // Per-cascade receiver culling planes (see shadowReceiverCullingPlanes).
  // Empty, or shorter than the cascades, for no receiver culling. Cached
  // static tiles outlive the frame's view, so they never use these.
  final List<List<Plane>> _cascadeReceiverPlanes;

  List<Plane> _receiverPlanesFor(int cascade) =>
      cascade < _cascadeReceiverPlanes.length
      ? _cascadeReceiverPlanes[cascade]
      : const [];
  final int _tileResolution;

  // Edge of a spot or point tile; below [_tileResolution] they stack in
  // columns after the cascades (Scene.punctualShadowsAtOwnResolution).
  final int _punctualTileResolution;
  final ShadowCasterFaces _casterFaces;

  // The directional light's shadow-caster channels. Applies to the cascade
  // tiles only; spot and point tiles use their own light's mask.
  final int _casterChannelMask;
  final SpotShadowFrame? _spotShadows;
  final PointShadowFrame? _pointShadows;
  final ShadowCachePlan? _cachePlan;
  final SharedShadowAtlas? _shared;

  Vector3 get _casterCameraPosition =>
      _shared?.cameraPosition ?? _cameraPosition;

  // The packed PostShadowInfo block, published for depth-aware custom passes.
  final ByteData? _shadowUniform;

  // Bound as FrameInfo.camera_position so a `vertex { }` material's
  // camera-relative displacement bends shadow casters the same way as the
  // color pass (see ShadowEncoder).
  final Vector3 _cameraPosition;

  static final gpu.Shader _copyVertexShader =
      baseShaderLibrary['FullscreenVertex']!;
  static final gpu.Shader _copyFragmentShader =
      baseShaderLibrary['ShadowCopyFragment']!;

  // Two triangles of NDC positions covering the viewport (6 vec2s).
  static final gpu.DeviceBuffer _quadBuffer = gpu.gpuContext
      .createDeviceBufferWithCopy(
        ByteData.sublistView(
          Float32List.fromList(<double>[
            -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, //
            -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, //
          ]),
        ),
      );
  static final gpu.BufferView _quadView = gpu.BufferView(
    _quadBuffer,
    offsetInBytes: 0,
    lengthInBytes: 6 * 2 * 4,
  );

  @override
  String get name => 'ShadowPass';

  @override
  void execute(RenderGraphContext context) {
    final shared = _shared;
    final sharedAtlas = shared?.atlas;
    if (sharedAtlas != null) {
      context.blackboard.set(kShadowMapBlackboardKey, sharedAtlas);
      final info = shared!.spotInfo;
      if (info != null) {
        context.blackboard.set(kSpotShadowInfoBlackboardKey, info);
      }
      return;
    }
    final deferredBefore = deferredPipelineBuilds;
    final plan = _cachePlan;
    final spotCount = _spotShadows?.matrices.length ?? 0;
    final pointCount = _pointShadows?.casters.length ?? 0;
    // No caster at all: every texel of the atlas would hold the clear value
    // (lit), which a white placeholder reads the same, so neither the atlas
    // nor the cached tiles are allocated. The tiles a plan asked to refresh
    // render once a caster appears.
    if (spotCount == 0 &&
        pointCount == 0 &&
        !debugAlwaysAllocateShadowAtlas &&
        !_hasCascadeCasters()) {
      if (plan != null) {
        for (final refresh in plan.refreshes) {
          refresh.entry.hasContent = false;
        }
      }
      context.blackboard.set(
        kShadowMapBlackboardKey,
        Material.whitePlaceholder(null),
      );
      final shadowUniform = _shadowUniform;
      if (shadowUniform != null) {
        context.blackboard.set(kShadowUniformBlackboardKey, shadowUniform);
      }
      return;
    }
    if (plan != null) {
      _renderStaticTiles(context, plan);
    }
    // WR-77: a site shadow map with no dynamic caster, spot, or point tile is
    // exactly its one cached tile, so that tile is published as the atlas:
    // no per-frame clear or tile copy (it re-renders only when the sun turns
    // or the static casters change).
    if (_siteShadow &&
        plan != null &&
        _cascades.length == 1 &&
        spotCount == 0 &&
        pointCount == 0 &&
        !debugDisableSiteShadowTilePublish &&
        !_hasDynamicCascadeCasters()) {
      context.blackboard.set(kShadowMapBlackboardKey, plan.entries[0].tile!);
      final shadowUniform = _shadowUniform;
      if (shadowUniform != null) {
        context.blackboard.set(kShadowUniformBlackboardKey, shadowUniform);
      }
      return;
    }
    final lampTiles = spotCount + pointCount * kPointShadowTilesPerLight;
    final totalTiles = _cascades.length + lampTiles;
    final lampResolution = _punctualTileResolution;
    final lampsPerColumn = _tileResolution ~/ lampResolution;
    final lampColumns = (lampTiles + lampsPerColumn - 1) ~/ lampsPerColumn;
    final atlasWidth =
        _tileResolution * _cascades.length + lampResolution * lampColumns;
    // Cached cascades without a dynamic caster, and no spot or point tile,
    // only replay their tiles, which needs no depth: the atlas then renders
    // without an atlas-sized depth attachment.
    final dynamicCasters = plan != null && _hasDynamicCascadeCasters();
    final needsDepth = plan == null || dynamicCasters || lampTiles > 0;
    // The atlas does not follow the output size, so it lives in the view's
    // size-independent pool: a render scale that toggles keeps one copy.
    final pool = context.texturePool.sizeIndependent;
    // fp32 (not fp16): the far cascade's orthographic depth range spans
    // hundreds of world units, and fp16's ~11-bit mantissa quantizes
    // window-space depth into steps coarser than the shadow depth bias.
    // That made the flat distant ground self-shadow in moire bands.
    final color = pool.acquire(
      TransientTextureDescriptor.color(
        width: atlasWidth,
        height: _tileResolution,
        format: gpu.PixelFormat.r32Float,
        debugName: 'directional_shadow_map',
        // Backends cache a framebuffer per color texture, so the copy-only
        // atlas keeps its own texture.
        attachmentKey: needsDepth ? null : 'no_depth',
      ),
    );
    final depth = needsDepth
        ? pool.acquire(
            TransientTextureDescriptor.depth(
              width: atlasWidth,
              height: _tileResolution,
              format: gpu.gpuContext.defaultDepthStencilFormat,
              debugName: 'directional_shadow_map_depth',
            ),
          )
        : null;
    final target = gpu.RenderTarget.singleColor(
      gpu.ColorAttachment(
        texture: color,
        // White = depth 1.0 in .r => fragments no caster covers are lit.
        clearValue: Vector4(1.0, 1.0, 1.0, 1.0),
      ),
      depthStencilAttachment: depth == null
          ? null
          : gpu.DepthStencilAttachment(texture: depth, depthClearValue: 1.0),
    );
    final commandBuffer = gpu.gpuContext.createCommandBuffer();
    final renderPass = commandBuffer.createRenderPass(target);

    // Each tile renders into its own slot of the strip. The viewport restricts
    // rasterization to the tile; the shared depth attachment is cleared once for
    // the whole atlas. Renders a tile from a light-space matrix.
    void renderTile(
      gpu.Viewport viewport,
      Matrix4 matrix,
      ShadowCasterFaces faces, {
      ShadowCasterFilter filter = ShadowCasterFilter.all,
      int casterChannelMask = 0xFF,
      List<Plane> receiverPlanes = const [],
      bool siteBias = false,
    }) {
      renderPass.setViewport(viewport);
      final encoder = ShadowEncoder(
        renderPass,
        context.transientsBuffer,
        matrix,
        _casterCameraPosition,
        faces,
        filter: filter,
        casterChannelMask: casterChannelMask,
        receiverPlanes: receiverPlanes,
        siteBias: siteBias,
        minCasterSize: siteBias ? _siteMinCasterSize : 0.0,
      );
      _renderScene.cull(
        encoder.frustum,
        encoder.submitCulled,
        additionalPlanes: receiverPlanes,
      );
      encoder.flush();
    }

    // Cascade tiles first (0..cascades.length), then the spot cones. With a
    // cache plan, each cascade tile replays its cached static content and
    // draws only the dynamic casters; otherwise everything renders. The
    // dynamic composite iterates the flat item list instead of culling the
    // spatial structure: dynamic casters are typically a handful, and four
    // per-cascade BVH traversals of a large static world cost far more than
    // one flag check per item (the encoder still frustum-culls each caster).
    for (var c = 0; c < _cascades.length; c++) {
      if (plan != null) {
        _encodeTileCopy(
          renderPass,
          c,
          plan.entries[c].tile!,
          writeDepth: needsDepth,
        );
        // Without a dynamic caster the composite draws nothing more.
        if (!dynamicCasters) continue;
        renderPass.setViewport(
          gpu.Viewport(
            x: c * _tileResolution,
            y: 0,
            width: _tileResolution,
            height: _tileResolution,
          ),
        );
        final encoder = ShadowEncoder(
          renderPass,
          context.transientsBuffer,
          _cascades[c].lightSpaceMatrix,
          _cameraPosition,
          _casterFaces,
          filter: ShadowCasterFilter.dynamicOnly,
          casterChannelMask: _casterChannelMask,
          receiverPlanes: _receiverPlanesFor(c),
          siteBias: _siteShadow,
          minCasterSize: _siteMinCasterSize,
        );
        for (final item in _renderScene.items) {
          encoder.submit(item);
        }
        encoder.flush();
      } else {
        renderTile(
          gpu.Viewport(
            x: c * _tileResolution,
            y: 0,
            width: _tileResolution,
            height: _tileResolution,
          ),
          _cascades[c].lightSpaceMatrix,
          _casterFaces,
          casterChannelMask: _casterChannelMask,
          receiverPlanes: _receiverPlanesFor(c),
          siteBias: _siteShadow,
        );
      }
    }
    // Spot and point tiles follow the cascades, `lampsPerColumn` to a column
    // of `lampResolution` (one per column when they match the cascades).
    final lampOriginX = _tileResolution * _cascades.length;
    int lampX(int slot) =>
        lampOriginX + (slot ~/ lampsPerColumn) * lampResolution;
    int lampY(int slot) => (slot % lampsPerColumn) * lampResolution;
    final spots = _spotShadows;
    if (spots != null) {
      for (var s = 0; s < spots.matrices.length; s++) {
        renderTile(
          gpu.Viewport(
            x: lampX(s),
            y: lampY(s),
            width: lampResolution,
            height: lampResolution,
          ),
          spots.matrices[s],
          spots.casterFaces,
          casterChannelMask: spots.casterChannelMasks[s],
        );
      }
    }
    final pointFrame = _pointShadows;
    if (pointFrame != null) {
      // Each caster's six 90-degree faces pack as 2x2 quadrants across two
      // tiles at half the tile edge, so a point light costs two atlas tiles
      // rather than six. The quadrant placement must stay in lockstep with the
      // shader's face -> atlas mapping (SamplePointShadow).
      final half = lampResolution ~/ 2;
      for (var s = 0; s < pointFrame.casters.length; s++) {
        for (var f = 0; f < kPointShadowFaces; f++) {
          final slot = spotCount + s * kPointShadowTilesPerLight + f ~/ 4;
          final q = f % 4;
          renderPass.setViewport(
            gpu.Viewport(
              x: lampX(slot) + (q & 1) * half,
              y: lampY(slot) + (q >> 1) * half,
              width: half,
              height: half,
            ),
          );
          final encoder = ShadowEncoder(
            renderPass,
            context.transientsBuffer,
            pointFrame.faceMatrix(s, f),
            _casterCameraPosition,
            pointFrame.casterFaces,
            casterChannelMask: pointFrame.casterChannelMasks[s],
          );
          _renderScene.cull(encoder.frustum, encoder.submitCulled);
          encoder.flush();
        }
      }
    }

    rendererSubmissions.submit(commandBuffer);
    context.blackboard.set(kShadowMapBlackboardKey, color);
    final spotFrame = _spotShadows;
    SpotShadowInfo? spotInfo;
    if (spotFrame != null) {
      spotInfo = SpotShadowInfo(
        matrices: spotFrame.matrices,
        firstTile: _cascades.length,
        totalTiles: totalTiles,
        tileResolution: _tileResolution,
        lampTileResolution: lampResolution,
      );
      context.blackboard.set(kSpotShadowInfoBlackboardKey, spotInfo);
    }
    // The pooled texture stays intact for the rest of the frame: its pool is
    // this view's, which renders nothing else until its next frame, and the
    // GPU runs the frame's command buffers in submission order.
    // An atlas missing deferred draws is not shared, so every later view
    // renders its own and counts its own deferred draws.
    if (shared != null && deferredPipelineBuilds == deferredBefore) {
      shared
        ..atlas = color
        ..spotInfo = spotInfo;
    }
    final shadowUniform = _shadowUniform;
    if (shadowUniform != null) {
      context.blackboard.set(kShadowUniformBlackboardKey, shadowUniform);
    }
  }

  /// Renders the plan's refresh jobs: static casters into the persistent
  /// per-cascade tiles. One command buffer per tile (a command buffer holds
  /// a single render pass encoder), submitted before the atlas buffer so the
  /// composite below reads the fresh content.
  void _renderStaticTiles(RenderGraphContext context, ShadowCachePlan plan) {
    for (final refresh in plan.refreshes) {
      final commandBuffer = gpu.gpuContext.createCommandBuffer();
      final entry = refresh.entry;
      entry.tile ??= gpu.gpuContext.createTexture(
        gpu.StorageMode.devicePrivate,
        _tileResolution,
        _tileResolution,
        format: gpu.PixelFormat.r32Float,
      );
      // Every refresh clears it, so one texture serves all the tiles.
      final depth = plan.cache.tileDepth ??= gpu.gpuContext.createTexture(
        gpu.StorageMode.deviceTransient,
        _tileResolution,
        _tileResolution,
        format: gpu.gpuContext.defaultDepthStencilFormat,
        enableRenderTargetUsage: true,
        enableShaderReadUsage: false,
      );
      final target = gpu.RenderTarget.singleColor(
        gpu.ColorAttachment(
          texture: entry.tile!,
          clearValue: Vector4(1.0, 1.0, 1.0, 1.0),
        ),
        depthStencilAttachment: gpu.DepthStencilAttachment(
          texture: depth,
          depthClearValue: 1.0,
        ),
      );
      final renderPass = commandBuffer.createRenderPass(target);
      final encoder = ShadowEncoder(
        renderPass,
        context.transientsBuffer,
        entry.matrix,
        _cameraPosition,
        _casterFaces,
        filter: ShadowCasterFilter.staticOnly,
        casterChannelMask: _casterChannelMask,
        siteBias: _siteShadow,
        minCasterSize: _siteMinCasterSize,
      );
      final deferred = deferredPipelineBuilds;
      _renderScene.cull(encoder.frustum, encoder.submitCulled);
      encoder.flush();
      rendererSubmissions.submit(commandBuffer);
      // A sliced warm-up skipped some of its casters, so render it again on
      // the next frame rather than reuse it.
      if (deferredPipelineBuilds > deferred) entry.hasContent = false;
    }
  }

  /// Whether any item would draw as a caster, static or dynamic, into a
  /// cascade tile (the view-independent test).
  bool _hasCascadeCasters() {
    if (_cascades.isEmpty) return false;
    for (final item in _renderScene.items) {
      if (shadowCasterAccepted(
        item,
        ShadowCasterFilter.all,
        _casterChannelMask,
      )) {
        return true;
      }
    }
    return false;
  }

  /// Whether any item would draw as a dynamic caster into a cascade tile
  /// (the view-independent test; culling only removes more).
  bool _hasDynamicCascadeCasters() {
    for (final item in _renderScene.items) {
      if (shadowCasterAccepted(
        item,
        ShadowCasterFilter.dynamicOnly,
        _casterChannelMask,
      )) {
        return true;
      }
    }
    return false;
  }

  /// Replays cascade [tile]'s cached static content into its atlas slot,
  /// writing the stored depth to both the color channel and the fragment
  /// depth so the dynamic casters drawn after depth-test against it.
  void _encodeTileCopy(
    gpu.RenderPass pass,
    int tile,
    gpu.Texture source, {
    bool writeDepth = true,
  }) {
    pass.setViewport(
      gpu.Viewport(
        x: tile * _tileResolution,
        y: 0,
        width: _tileResolution,
        height: _tileResolution,
      ),
    );
    pass.clearBindings();
    final pipeline = resolvePipeline(_copyVertexShader, _copyFragmentShader);
    pass.bindPipeline(pipeline);
    pass.setColorBlendEnable(false);
    pass.setCullMode(gpu.CullMode.none);
    pass.setDepthWriteEnable(writeDepth);
    pass.setDepthCompareOperation(gpu.CompareFunction.always);
    bindVertexBufferCompat(pass, _quadView, 6);
    pass.bindTexture(
      _copyFragmentShader.cachedUniformSlot('source_texture'),
      source,
    );
    drawCompat(pass, 6);
  }
}
