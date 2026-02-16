#version 460 core

uniform sampler2D dst;
uniform sampler2D src;
uniform int channel;       // 0=R, 1=G, 2=B, 3=A
uniform float strength;
uniform vec4 region;       // x, y, w, h in UV coordinates [0, 1]

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 dst_color = texture(dst, texCoord);

    // Check if texCoord is within region
    vec2 regionMin = region.xy;
    vec2 regionMax = region.xy + region.zw;

    if (texCoord.x >= regionMin.x && texCoord.x < regionMax.x &&
        texCoord.y >= regionMin.y && texCoord.y < regionMax.y) {
        // Map texCoord within region to full [0,1] UV for source
        vec2 srcUV = (texCoord - regionMin) / region.zw;
        float src_val = texture(src, srcUV).r * strength;

        // Replace target channel only
        if (channel == 0) dst_color.r = src_val;
        else if (channel == 1) dst_color.g = src_val;
        else if (channel == 2) dst_color.b = src_val;
        else if (channel == 3) dst_color.a = src_val;
    }

    fragColor = dst_color;
}
