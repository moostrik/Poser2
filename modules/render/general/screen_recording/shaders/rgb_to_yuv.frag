#version 460 core

#include "position.glsl"

// One YUV plane of the RGB screen frame, BT.709 video range, the picture's top row first, so the plane reads back
// in the order a video file holds it. plane 0: Y, at the frame's size; 1: Cb Cr interleaved at half width and half
// height (4:2:0), each sample the average of its 2 x 2 source pixels; 2: Cb and 3: Cr at the frame's size (4:4:4).

uniform sampler2D src;
uniform int plane;
uniform mat3 matrix;    // rows: Y, Cb, Cr coefficients on RGB 0..1, scaled to video range
uniform vec3 offset;    // 16, 128, 128 over 255

in vec2 texCoord;
out vec4 fragColor;

vec3 ycbcr(vec3 rgb) {
    return vec3(dot(matrix[0], rgb), dot(matrix[1], rgb), dot(matrix[2], rgb)) + offset;
}

void main() {
    vec2 at = topLeft(texCoord) * vec2(textureSize(src, 0));     // the source's texel under this one, top row first
    if (plane == 1) {
        ivec2 q = ivec2(floor(at - 0.5));
        vec3 rgb = 0.25 * (texelFetch(src, q, 0).rgb + texelFetch(src, q + ivec2(1, 0), 0).rgb
                         + texelFetch(src, q + ivec2(0, 1), 0).rgb + texelFetch(src, q + ivec2(1, 1), 0).rgb);
        vec3 c = ycbcr(rgb);
        fragColor = vec4(c.y, c.z, 0.0, 1.0);
    } else {
        vec3 c = ycbcr(texelFetch(src, ivec2(at), 0).rgb);
        fragColor = vec4(plane == 0 ? c.x : (plane == 2 ? c.y : c.z), 0.0, 0.0, 1.0);
    }
}
