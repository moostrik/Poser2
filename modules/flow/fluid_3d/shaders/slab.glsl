// The injection's slab: the body's gaussian profile in depth, its sigma uSpread, a share of the box's depth,
// fullest at each pixel's z position (`slab_center`), `#include "slab.glsl"`. A cell takes the share of the
// body that lies in it, the profile's area between its two faces over its length, not the profile at its
// middle: a body thinner than a cell fills the part of the cell it covers, moves smoothly between cells, and
// comes to the same amount at any number of them; a sigma of 0 puts nothing in.
//
// Needs, declared before: uniform float uTargetLayer (where a pixel without a z position goes, 0 the first
// layer's middle, 1 the last's), uniform float uSpread (the sigma, a share of the depth), uniform sampler2D
// uZPosition (a z position: red the z position times its weight, green the weight) and uniform bool
// uHasZPosition.

const float Z_POSITION_FLOOR = 1e-6;    // a weight below it is no z position

// The slab's center at `uv`: the z position there (red over green), uTargetLayer where it has none
float slab_center(vec2 uv) {
    if (!uHasZPosition) return uTargetLayer;
    vec2 z_position = texture(uZPosition, uv).rg;
    return z_position.g > Z_POSITION_FLOOR ? clamp(z_position.r / z_position.g, 0.0, 1.0) : uTargetLayer;
}

// erf, Abramowitz and Stegun 7.1.26 (its error under 1.5e-7)
float erf_of(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return sign(x) * y;
}

// The profile's middle in the depth's own share (0 the front face), for `center` placed as the layers place it
float slab_middle(float center, int layers) {
    return (0.5 + center * (float(layers) - 1.0)) / float(layers);
}

// The body's share of the depth from `front` to `back` (shares of the depth, 0 the front), over that length:
// the profile's mean there, 1 at the middle of a thick body
float slab_between(float front, float back, float center, int layers) {
    if (uSpread <= 1e-6) return 0.0;
    float middle = slab_middle(center, layers);
    float scale = 1.0 / (uSpread * 1.41421356);
    float area = 0.5 * uSpread * 2.50662827 * (erf_of((back - middle) * scale) - erf_of((front - middle) * scale));
    return area / max(back - front, 1e-9);
}

// A cell's share, layer `z` of `layers`
float slab_cell(int z, float center, int layers) {
    return slab_between(float(z) / float(layers), float(z + 1) / float(layers), center, layers);
}

// The share over one layer's length centered at `z` (cells): for a face, between the cells it joins
float slab_at(float z, float center, int layers) {
    return slab_between((z - 0.5) / float(layers), (z + 0.5) / float(layers), center, layers);
}

// The cells a body at `center` reaches with a share worth adding, [first, last]: its profile out to 4 sigmas
ivec2 slab_cells(float center, int layers) {
    float middle = slab_middle(center, layers) * float(layers);
    float reach = 4.0 * uSpread * float(layers) + 1.0;
    return ivec2(clamp(int(floor(middle - reach)), 0, layers - 1), clamp(int(ceil(middle + reach)), 0, layers - 1));
}

// The shares summed over the grid's layers: what one unit a step puts into a column (the particles' mass)
float slab_column(float center, int layers) {
    if (uSpread <= 1e-6) return 0.0;
    float middle = slab_middle(center, layers);
    float scale = 1.0 / (uSpread * 1.41421356);
    float area = 0.5 * uSpread * 2.50662827 * (erf_of((1.0 - middle) * scale) - erf_of(-middle * scale));
    return area * float(layers);
}
