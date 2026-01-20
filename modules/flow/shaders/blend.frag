#version 460 core

uniform sampler2D dst;
uniform sampler2D src;
uniform float dst_strength;
uniform float src_strength;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 dst_color = texture(dst, texCoord) * dst_strength;
    vec4 src_color = texture(src, texCoord) * src_strength;

    fragColor = dst_color + src_color;
}
