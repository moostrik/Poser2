// The density's blend, `#include "density_blend.glsl"`: what a pixel of the bridge brings and how it goes into a
// cell, the same for the density, the fine density's amount and the particles. Premultiplied throughout: a
// density is its color times its amount, and its amount.
//   OVER  compositing over (Porter and Duff 1984): the cell becomes the source plus the cell × (1 − the source's
//         amount), by the share; the amount never falls and never passes 1, the color is the amount-weighted mix
//   ADD   a source (Stam 2003): the source × the share added; where the amount would pass 1, all four channels
//         are scaled down together so it lands at 1

const int BLEND_OVER = 0;
const int BLEND_ADD = 1;

uniform int   uBlend;       // BLEND_OVER or BLEND_ADD
uniform float uGain;        // the bridge's amount times this

// The amount a pixel brings, from the bridge's: times the gain, capped at 1 over
float density_amount(float amount) {
    float a = abs(amount) * uGain;
    return uBlend == BLEND_OVER ? min(a, 1.0) : a;
}

// `cell` with `source` blended in by `share`: over, the drag's share of the way; add, the share of the source
vec4 blend_density(vec4 cell, vec4 source, float share) {
    if (uBlend == BLEND_OVER) return cell + (source - cell * source.a) * min(share, 1.0);
    vec4 sum = cell + source * share;
    return sum.a > 1.0 ? sum / sum.a : sum;
}
