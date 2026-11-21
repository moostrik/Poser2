#version 460 core

#include "position.glsl"

// RGB from a screen frame's YUV planes in BT.709 video range, as `RgbToYuv` writes them and uploaded with the
// picture's top row first, so as GL textures they hold the picture upside down; the target gets it upright.
// 1 plane: Cb, Y, Cr in red, green, blue (4:4:4 packed); 2: Y, and Cb Cr interleaved at half size (4:2:0);
// 3: Y, Cb and Cr (4:4:4).

layout(binding = 0) uniform sampler2D plane0;
layout(binding = 1) uniform sampler2D plane1;
layout(binding = 2) uniform sampler2D plane2;
uniform int num_planes;
uniform mat3 matrix;            // rows: R, G, B coefficients on Y Cb Cr 0..1, the inverse of `RgbToYuv`'s
uniform vec3 offset;            // 16, 128, 128 as normalized samples of the planes

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 coord = topLeft(texCoord);     // the target's position from the top left is the frame's row order
    vec3 ycbcr;
    if (num_planes == 1) {
        ycbcr = texture(plane0, coord).grb;
    } else if (num_planes == 2) {
        ycbcr = vec3(texture(plane0, coord).r, texture(plane1, coord).rg);
    } else {
        ycbcr = vec3(texture(plane0, coord).r, texture(plane1, coord).r, texture(plane2, coord).r);
    }
    ycbcr -= offset;
    vec3 rgb = vec3(dot(matrix[0], ycbcr), dot(matrix[1], ycbcr), dot(matrix[2], ycbcr));
    fragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}
