#version 460 core

// Inject Density 2D Shader
// The bridge's density into the 2D density, in place of the density before (uDensity): the source is its color
// (premultiplied by its alpha, the coverage) times its amount (density_blend.glsl), blended into each texel by
// uStrength (× uRelease where it falls); with uOklab the source turned into Oklab first, (L, a, b) × its amount.

uniform sampler2D uDensity;    // the density before (RGBA, premultiplied)
uniform sampler2D uAmount;     // the bridge's amount, red
uniform sampler2D uColor;      // its color, premultiplied by its alpha
uniform float uStrength;       // over, fill: the drag's share this step; add: dt
uniform float uRelease;        // the release's share this step where the texel falls; 0 = never falls
uniform bool  uOklab;          // the density holds Oklab

in vec2 texCoord;
out vec4 fragColor;

#include "color.glsl"
#include "../../shaders/density_blend.glsl"

void main() {
    vec4 cell = texture(uDensity, texCoord);
    vec4 color = texture(uColor, texCoord);
    float amount = texture(uAmount, texCoord).r;
    vec4 source = color * density_amount(amount);
    float level = density_level(amount);     // over and add: what the release falls toward
    if (source.a <= 0.0) {
        fragColor = cell;
        return;
    }
    if (uOklab) source.rgb = rgb2oklab(source.rgb / source.a) * source.a;
    float release = release_share(uRelease, color.a);
    fragColor = blend_density(cell, source, level, uStrength, release);
}
