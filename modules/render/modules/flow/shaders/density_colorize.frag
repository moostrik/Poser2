#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec4 fragColor;

// Input textures
uniform sampler2D density;     // Density field (RGBA16F) - each channel is a different track

// Track colors - up to 4 tracks mapped to RGBA channels
uniform vec4 colors[4];

#include "color.glsl"

void main() {
    vec4 den = texture(density, texCoord);

    // Map each density channel to its corresponding track color
    // R channel -> track 0, G -> track 1, B -> track 2, A -> track 3
    vec3 result = vec3(0.0);
    result += den.r * colors[0].rgb;
    result += den.g * colors[1].rgb;
    result += den.b * colors[2].rgb;
    result += den.a * colors[3].rgb;

    // Normalize brightness using HSV (clamp V to 1.0 if overbright)
    vec3 hsv = rgb2hsv(result);
    if (hsv.z > 1.0) {
        hsv.z = 1.0;
        result = hsv2rgb(hsv);
    }

    // Alpha from max of densities (for transparency blending)
    float alpha = max(max(den.r, den.g), max(den.b, den.a));

    fragColor = vec4(result, alpha);
}
