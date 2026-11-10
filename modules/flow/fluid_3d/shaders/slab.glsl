// The injection's slab: the body's gaussian profile in depth, fullest at uTargetLayer, its sigma uSpread, both
// shares of the box's depth (`#include "slab.glsl"`). A cell takes the share of the body that lies in it, the
// profile's area between its two faces over its length, not the profile at its middle: a body thinner than a
// cell fills the part of the cell it covers, moves smoothly between cells, and comes to the same amount at any
// number of them; a sigma of 0 puts nothing in.
//
// Needs, declared before: uniform float uTargetLayer (0 the first layer's middle, 1 the last's), uniform float
// uSpread (the sigma, a share of the depth).

// erf, Abramowitz and Stegun 7.1.26 (its error under 1.5e-7)
float erf_of(float x) {
    float t = 1.0 / (1.0 + 0.3275911 * abs(x));
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return sign(x) * y;
}

// The body's share of the depth from `front` to `back` (shares of the depth, 0 the front), over that length:
// the profile's mean there, 1 at the middle of a thick body
float slab_between(float front, float back, int layers) {
    if (uSpread <= 1e-6) return 0.0;
    float target = (0.5 + uTargetLayer * (float(layers) - 1.0)) / float(layers);
    float scale = 1.0 / (uSpread * 1.41421356);
    float area = 0.5 * uSpread * 2.50662827 * (erf_of((back - target) * scale) - erf_of((front - target) * scale));
    return area / max(back - front, 1e-9);
}

// A cell's share, layer `z` of `layers`
float slab_cell(int z, int layers) {
    return slab_between(float(z) / float(layers), float(z + 1) / float(layers), layers);
}

// The share over one layer's length centered at `z` (cells): for a face, between the cells it joins
float slab_at(float z, int layers) {
    return slab_between((z - 0.5) / float(layers), (z + 0.5) / float(layers), layers);
}
