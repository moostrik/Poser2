#version 460 core

// Density To RGB 2D Shader
// The 2D density as the picture: held in Oklab, (L, a, b) × its amount and its amount, turned back to RGB divided by
// its amount, turned, multiplied again, premultiplied as it was; held in RGB, as it is.

layout(binding = 0) uniform sampler2D uDensity;   // the density (premultiplied)
uniform bool uOklab;           // the density holds Oklab

in vec2 texCoord;
out vec4 fragColor;

#include "color.glsl"

void main() {
    vec4 density = texture(uDensity, texCoord);
    if (uOklab && density.a > 0.0) density.rgb = max(oklab2rgb(density.rgb / density.a), vec3(0.0)) * density.a;
    fragColor = density;
}
