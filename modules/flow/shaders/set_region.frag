#version 460 core

#include "position.glsl"

uniform sampler2D dst;
uniform sampler2D src;
uniform float strength;
uniform vec4 region;  // x, y, w, h in 0..1 of the target, origin top-left

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 dst_color = texture(dst, texCoord);

    // Check if texCoord is within region
    vec2 regionMin = region.xy;
    vec2 regionMax = region.xy + region.zw;

    vec2 p = topLeft(texCoord);
    if (p.x >= regionMin.x && p.x < regionMax.x &&
        p.y >= regionMin.y && p.y < regionMax.y) {
        // The point within the region, over the whole source
        vec2 srcUV = topLeft((p - regionMin) / region.zw);
        fragColor = texture(src, srcUV) * strength;
    } else {
        fragColor = dst_color;
    }
}
