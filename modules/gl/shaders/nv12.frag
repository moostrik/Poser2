#version 460 core

// One plane of NV12 from an RGB texture of the recording's size, in BT.709 video range,
// flipped so that the first row read back is the picture's top row.
// plane 0: Y, one channel, at the source's size; plane 1: UV (Cb, Cr), two channels, at half
// width and half height, each sample the average of its 2 x 2 source pixels.

uniform sampler2D src;
uniform int plane;
uniform mat3 matrix;    // rows: Y, Cb, Cr coefficients on RGB 0..1, scaled to video range
uniform vec3 offset;    // 16, 128, 128 over 255

out vec4 fragColor;

vec3 source(ivec2 p, int height) {
    return texelFetch(src, ivec2(p.x, height - 1 - p.y), 0).rgb;
}

void main() {
    ivec2 size = textureSize(src, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (plane == 0) {
        vec3 rgb = source(p, size.y);
        fragColor = vec4(dot(matrix[0], rgb) + offset.x, 0.0, 0.0, 1.0);
    } else {
        ivec2 q = p * 2;
        vec3 rgb = 0.25 * (source(q, size.y) + source(q + ivec2(1, 0), size.y)
                         + source(q + ivec2(0, 1), size.y) + source(q + ivec2(1, 1), size.y));
        fragColor = vec4(dot(matrix[1], rgb) + offset.y, dot(matrix[2], rgb) + offset.z, 0.0, 1.0);
    }
}
