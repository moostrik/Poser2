#version 460 core

uniform sampler2D src;
uniform sampler2D dst;
uniform sampler2D mask;
uniform float blend;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 src_color = texture(src, texCoord);
    vec4 dst_color = texture(dst, texCoord);
    float mask_value = texture(mask, texCoord).x;
    float mix_value = clamp((blend * 2.0 - 1.0) + pow(mask_value, 1.8), 0.0, 1.0);
    fragColor = mix(src_color, dst_color, mix_value);
}