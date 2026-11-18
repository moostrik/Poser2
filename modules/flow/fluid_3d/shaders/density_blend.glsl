// The density's blend, `#include "density_blend.glsl"`: what a pixel of the bridge brings and how it goes into a
// cell, the same for the density, the fine density's amount and the particles. Premultiplied throughout: a
// density is its color times its amount, and its amount.
//   OVER  compositing over (Porter and Duff 1984): the cell becomes the source plus the cell × (1 − the source's
//         amount), by the share; the amount never falls and never passes 1, the color is the amount-weighted mix
//   ADD   a source (Stam 2003): the source × the share added; where the amount would pass 1, all four channels
//         are scaled down together so it lands at 1
//   FILL  the cell drawn toward the source, its amount at most 1, its color kept: by the share where the source's
//         amount is above the cell's, else by the release's; never past it
// The release, over and add: where the cell's amount is above the pixel's level (density_level), all four
// channels scaled down together toward it by the release's share times the pixel's coverage (release_share), as
// over's share is weighted by the source's alpha, the cell's color kept; 0 = never falls

const int BLEND_OVER = 0;
const int BLEND_ADD = 1;
const int BLEND_FILL = 2;

uniform int   uBlend;       // BLEND_OVER, BLEND_ADD or BLEND_FILL
uniform float uGain;        // the bridge's amount times this

// The amount a pixel brings, from the bridge's: times the gain, capped at 1 over
float density_amount(float amount) {
    float a = abs(amount) * uGain;
    return uBlend == BLEND_OVER ? min(a, 1.0) : a;
}

// The level a pixel holds a cell down to by the release, over and add: the bridge's amount times the gain,
// capped at 1; the coverage weights the release's share, it is not a level
float density_level(float amount) {
    return min(abs(amount) * uGain, 1.0);
}

// The release's share in a cell, `release` the slab's: over and add, times the pixel's coverage
float release_share(float release, float coverage) {
    return uBlend == BLEND_FILL ? release : release * coverage;
}

// `cell` with `source` blended in by `share`: over, the drag's share of the way; add, the share of the source;
// fill, the drag's share up and `release` down, toward the source capped at fully dyed; over and add then fall
// toward `level` by `release` where the cell is above it
vec4 blend_density(vec4 cell, vec4 source, float level, float share, float release) {
    if (uBlend == BLEND_FILL) {
        vec4 target = source.a > 1.0 ? source / source.a : source;
        return cell + (target - cell) * min(target.a > cell.a ? share : release, 1.0);
    }
    vec4 blended;
    if (uBlend == BLEND_OVER) {
        blended = cell + (source - cell * source.a) * min(share, 1.0);
    } else {
        blended = cell + source * share;
        if (blended.a > 1.0) blended /= blended.a;
    }
    if (blended.a > level) blended *= 1.0 - min(release, 1.0) * (1.0 - level / blended.a);
    return blended;
}
