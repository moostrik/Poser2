#version 460 core

uniform sampler2D dst;
uniform sampler2D src;
uniform float src_strength;
uniform vec4 region;  // x, y, w, h in UV coordinates [0, 1]

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
        vec4 src_color = texture(src, srcUV) * src_strength;
        fragColor = dst_color + src_color;
    } else {
        fragColor = dst_color;
    }
}
