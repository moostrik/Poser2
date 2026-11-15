#version 460 core

uniform sampler2D src;
uniform float threshold;
uniform float strength;
uniform bool invert;
#define texelSize (1.0 / vec2(textureSize(src, 0)))

#include "color.glsl"

in vec2 texCoord;
out vec4 fragColor;

float lum(vec2 offset) {
    return luminance(texture(src, texCoord + offset * texelSize).rgb);
}

void main() {
    // Sample 3x3 neighborhood and convert to luminance
    float tl = lum(vec2(-1.0,  1.0));
    float t  = lum(vec2( 0.0,  1.0));
    float tr = lum(vec2( 1.0,  1.0));
    float l  = lum(vec2(-1.0,  0.0));
    float r  = lum(vec2( 1.0,  0.0));
    float bl = lum(vec2(-1.0, -1.0));
    float b  = lum(vec2( 0.0, -1.0));
    float br = lum(vec2( 1.0, -1.0));

    // Sobel kernels
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;

    // Edge magnitude
    float edge = sqrt(gx*gx + gy*gy) * strength;

    // Threshold with smoothstep for anti-aliased edges
    float line = smoothstep(threshold - 0.05, threshold + 0.05, edge);

    // Invert: black edges on white, or white edges on black
    if (invert) {
        line = 1.0 - line;
    }

    fragColor = vec4(vec3(line), 1.0);
}
