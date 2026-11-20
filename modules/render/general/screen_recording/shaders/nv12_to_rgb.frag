#version 460 core

#include "position.glsl"

// RGB from the two planes of an NV12 frame in BT.709 video range, as `Nv12` writes them: the Y plane at full size and
// the Cb Cr plane at half size, each uploaded with the picture's top row first, so as GL textures they hold the
// picture upside down; the target gets it upright.

uniform sampler2D luma;         // R8: Y
uniform sampler2D chroma;       // RG8: Cb, Cr
uniform mat3 matrix;            // rows: R, G, B coefficients on Y Cb Cr 0..1, the inverse of `Nv12`'s
uniform vec3 offset;            // 16, 128, 128 over 255

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 coord = topLeft(texCoord);     // the target's position from the top left is the frame's row order
    vec3 ycbcr = vec3(texture(luma, coord).r, texture(chroma, coord).rg) - offset;
    vec3 rgb = vec3(dot(matrix[0], ycbcr), dot(matrix[1], ycbcr), dot(matrix[2], ycbcr));
    fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
