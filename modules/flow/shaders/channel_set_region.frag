#version 460 core

#include "position.glsl"

uniform sampler2D dst;
uniform sampler2D src;
uniform int channel;       // 0=R, 1=G, 2=B, 3=A
uniform float strength;
uniform vec4 region;       // x, y, w, h in 0..1 of the target, origin top-left

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
        float src_val = texture(src, srcUV).r * strength;

        // Replace target channel only
        if (channel == 0) dst_color.r = src_val;
        else if (channel == 1) dst_color.g = src_val;
        else if (channel == 2) dst_color.b = src_val;
        else if (channel == 3) dst_color.a = src_val;
    }

    fragColor = dst_color;
}
