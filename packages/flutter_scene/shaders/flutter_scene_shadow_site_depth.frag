// Fragment shader for a site shadow map's caster pass (WR-77). Like
// flutter_scene_depth_only.frag, plus a slope-scaled caster bias: Flutter GPU
// has no rasterizer depth-bias state, so the depth written for the shadow test
// is pushed away from the light by 1.5x its slope per shadow texel plus a
// small constant, as a WebGPU depthBias 2 / depthBiasSlopeScale 1.5 pipeline
// would (approach from Abyssal Garden, MIT, Noniv). The site lookup then needs
// no normal-offset bias at the receiver.
#ifdef FLUTTER_SCENE_SITE_DEPTH_MASKED
#define FLUTTER_SCENE_NO_VIEW_INFO
#include <material_varyings.glsl>
#include <material_inputs.glsl>
#include <depth_mask.glsl>
#else
out vec4 frag_color;
#endif

void main() {
#ifdef FLUTTER_SCENE_SITE_DEPTH_MASKED
  ApplyDepthAlphaMask();
#endif
  highp float z = gl_FragCoord.z;
  highp float slope = max(abs(dFdx(z)), abs(dFdy(z)));
  frag_color = vec4(z + 1.5 * slope + 2.0 / 16777216.0, 0.0, 0.0, 1.0);
}
