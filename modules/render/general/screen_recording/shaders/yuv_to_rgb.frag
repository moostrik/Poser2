#version 460 core

#include "position.glsl"

// RGB from a screen frame's YUV planes in BT.709 video range, as `RgbToYuv` writes them and uploaded with the
// picture's top row first, so as GL textures they hold the picture upside down; the target gets it upright.
// 4:2:0: `chroma` holds Cb Cr interleaved at half size; 4:4:4 (`full_chroma`): `chroma` holds Cb and `cr` Cr.

uniform sampler2D luma;         // R8: Y
uniform sampler2D chroma;       // RG8 at half size: Cb Cr; or R8 at full size: Cb
uniform sampler2D cr;           // R8 at full size: Cr (4:4:4 only)
uniform bool full_chroma;
uniform mat3 matrix;            // rows: R, G, B coefficients on Y Cb Cr 0..1, the inverse of `RgbToYuv`'s
uniform vec3 offset;            // 16, 128, 128 over 255

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 coord = topLeft(texCoord);     // the target's position from the top left is the frame's row order
    vec2 cbcr = full_chroma ? vec2(texture(chroma, coord).r, texture(cr, coord).r) : texture(chroma, coord).rg;
    vec3 ycbcr = vec3(texture(luma, coord).r, cbcr) - offset;
    vec3 rgb = vec3(dot(matrix[0], ycbcr), dot(matrix[1], ycbcr), dot(matrix[2], ycbcr));
    fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
