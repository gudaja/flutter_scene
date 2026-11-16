// The standard lit fragment shader for a site shadow map (WR-77,
// DirectionalLight.shadowSiteBounds): one fixed orthographic shadow tile, so
// the directional lookup is a single matrix and a 1- or 4-tap compare, with no
// cascade selection, normal-offset bias, or distance fade compiled in. Same
// body as flutter_scene_standard.frag otherwise. Selected exactly when
// Lighting.siteShadow is true.
#define FLUTTER_SCENE_SITE_SHADOW
#include <flutter_scene_standard.frag>
