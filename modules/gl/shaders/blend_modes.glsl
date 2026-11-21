// Blend modes: how a layer `s` over a base `b` combines, per channel, all 0..1. The separable modes are the W3C
// Compositing spec's and Photoshop's; hue, saturation, color and luminosity are the spec's non-separable ones, on its
// luminosity weights (0.3, 0.59, 0.11). Reflect and glow are smile's.

float blendColorBurn(float b, float s)   { return b >= 1.0 ? 1.0 : s <= 0.0 ? 0.0 : 1.0 - min(1.0, (1.0 - b) / s); }
float blendColorDodge(float b, float s)  { return b <= 0.0 ? 0.0 : s >= 1.0 ? 1.0 : min(1.0, b / (1.0 - s)); }
float blendSoftLightD(float b)           { return b <= 0.25 ? ((16.0 * b - 12.0) * b + 4.0) * b : sqrt(b); }
float blendSoftLight(float b, float s)   { return s <= 0.5 ? b - (1.0 - 2.0 * s) * b * (1.0 - b)
                                                           : b + (2.0 * s - 1.0) * (blendSoftLightD(b) - b); }
float blendOverlay(float b, float s)     { return b < 0.5 ? 2.0 * b * s : 1.0 - 2.0 * (1.0 - b) * (1.0 - s); }
float blendVividLight(float b, float s)  { return s < 0.5 ? blendColorBurn(b, 2.0 * s) : blendColorDodge(b, 2.0 * s - 1.0); }
float blendPinLight(float b, float s)    { return s < 0.5 ? min(b, 2.0 * s) : max(b, 2.0 * s - 1.0); }
float blendReflect(float b, float s)     { return s >= 1.0 ? 1.0 : min(1.0, b * b / (1.0 - s)); }

vec3 blendColorBurn(vec3 b, vec3 s)  { return vec3(blendColorBurn(b.r, s.r), blendColorBurn(b.g, s.g), blendColorBurn(b.b, s.b)); }
vec3 blendColorDodge(vec3 b, vec3 s) { return vec3(blendColorDodge(b.r, s.r), blendColorDodge(b.g, s.g), blendColorDodge(b.b, s.b)); }
vec3 blendSoftLight(vec3 b, vec3 s)  { return vec3(blendSoftLight(b.r, s.r), blendSoftLight(b.g, s.g), blendSoftLight(b.b, s.b)); }
vec3 blendOverlay(vec3 b, vec3 s)    { return vec3(blendOverlay(b.r, s.r), blendOverlay(b.g, s.g), blendOverlay(b.b, s.b)); }
vec3 blendVividLight(vec3 b, vec3 s) { return vec3(blendVividLight(b.r, s.r), blendVividLight(b.g, s.g), blendVividLight(b.b, s.b)); }
vec3 blendPinLight(vec3 b, vec3 s)   { return vec3(blendPinLight(b.r, s.r), blendPinLight(b.g, s.g), blendPinLight(b.b, s.b)); }
vec3 blendReflect(vec3 b, vec3 s)    { return vec3(blendReflect(b.r, s.r), blendReflect(b.g, s.g), blendReflect(b.b, s.b)); }

// The non-separable modes' helpers (W3C Compositing, "Blending")
float blendLum(vec3 c) { return dot(c, vec3(0.3, 0.59, 0.11)); }
vec3 blendClipColor(vec3 c) {
    float l = blendLum(c);
    float n = min(c.r, min(c.g, c.b));
    float x = max(c.r, max(c.g, c.b));
    if (n < 0.0) c = l + (c - l) * l / (l - n);
    if (x > 1.0) c = l + (c - l) * (1.0 - l) / (x - l);
    return c;
}
vec3 blendSetLum(vec3 c, float l) { return blendClipColor(c + (l - blendLum(c))); }
float blendSat(vec3 c) { return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b)); }
vec3 blendSetSat(vec3 c, float s) {
    float lo = min(c.r, min(c.g, c.b));
    float hi = max(c.r, max(c.g, c.b));
    return hi > lo ? (c - lo) * s / (hi - lo) : vec3(0.0);
}

// The mode numbers, as `gl.BlendState.BlendMode` numbers them
const int BLEND_NORMAL = 0;
const int BLEND_ADD = 1;
const int BLEND_MULTIPLY = 2;
const int BLEND_SCREEN = 3;
const int BLEND_DARKEN = 10;
const int BLEND_COLOR_BURN = 11;
const int BLEND_LINEAR_BURN = 12;
const int BLEND_LIGHTEN = 13;
const int BLEND_COLOR_DODGE = 14;
const int BLEND_OVERLAY = 15;
const int BLEND_SOFT_LIGHT = 16;
const int BLEND_HARD_LIGHT = 17;
const int BLEND_VIVID_LIGHT = 18;
const int BLEND_LINEAR_LIGHT = 19;
const int BLEND_PIN_LIGHT = 20;
const int BLEND_HARD_MIX = 21;
const int BLEND_DIFFERENCE = 22;
const int BLEND_EXCLUSION = 23;
const int BLEND_SUBTRACT = 24;
const int BLEND_DIVIDE = 25;
const int BLEND_HUE = 26;
const int BLEND_SATURATION = 27;
const int BLEND_COLOR = 28;
const int BLEND_LUMINOSITY = 29;
const int BLEND_REFLECT = 30;
const int BLEND_GLOW = 31;

// The layer `s` over the base `b` in `mode`; a number this does not know draws `s` as it is.
vec3 blendMode(int mode, vec3 b, vec3 s) {
    if (mode == BLEND_ADD)          return min(b + s, 1.0);
    if (mode == BLEND_MULTIPLY)     return b * s;
    if (mode == BLEND_SCREEN)       return 1.0 - (1.0 - b) * (1.0 - s);
    if (mode == BLEND_DARKEN)       return min(b, s);
    if (mode == BLEND_COLOR_BURN)   return blendColorBurn(b, s);
    if (mode == BLEND_LINEAR_BURN)  return max(b + s - 1.0, 0.0);
    if (mode == BLEND_LIGHTEN)      return max(b, s);
    if (mode == BLEND_COLOR_DODGE)  return blendColorDodge(b, s);
    if (mode == BLEND_OVERLAY)      return blendOverlay(b, s);
    if (mode == BLEND_SOFT_LIGHT)   return blendSoftLight(b, s);
    if (mode == BLEND_HARD_LIGHT)   return blendOverlay(s, b);
    if (mode == BLEND_VIVID_LIGHT)  return blendVividLight(b, s);
    if (mode == BLEND_LINEAR_LIGHT) return clamp(b + 2.0 * s - 1.0, 0.0, 1.0);
    if (mode == BLEND_PIN_LIGHT)    return blendPinLight(b, s);
    if (mode == BLEND_HARD_MIX)     return step(0.5, blendVividLight(b, s));
    if (mode == BLEND_DIFFERENCE)   return abs(b - s);
    if (mode == BLEND_EXCLUSION)    return b + s - 2.0 * b * s;
    if (mode == BLEND_SUBTRACT)     return max(b - s, 0.0);
    if (mode == BLEND_DIVIDE)       return vec3(s.r > 0.0 ? min(b.r / s.r, 1.0) : 1.0,
                                                s.g > 0.0 ? min(b.g / s.g, 1.0) : 1.0,
                                                s.b > 0.0 ? min(b.b / s.b, 1.0) : 1.0);
    if (mode == BLEND_HUE)          return blendSetLum(blendSetSat(s, blendSat(b)), blendLum(b));
    if (mode == BLEND_SATURATION)   return blendSetLum(blendSetSat(b, blendSat(s)), blendLum(b));
    if (mode == BLEND_COLOR)        return blendSetLum(s, blendLum(b));
    if (mode == BLEND_LUMINOSITY)   return blendSetLum(b, blendLum(s));
    if (mode == BLEND_REFLECT)      return blendReflect(b, s);
    if (mode == BLEND_GLOW)         return blendReflect(s, b);
    return s;
}
