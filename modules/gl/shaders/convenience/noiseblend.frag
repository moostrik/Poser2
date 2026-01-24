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
    vec3 mask_value = texture(mask, texCoord).xyz;
    vec4 mix_value = vec4(1);
    if (mask_value.r > blend) mix_value.r = 0;
    if (mask_value.g > blend) mix_value.g = 0;
    if (mask_value.b > blend) mix_value.b = 0;

    fragColor = vec4(1.0);
    fragColor.r = mix(src_color.r, dst_color.r, mix_value.r);
    fragColor.g = mix(src_color.g, dst_color.g, mix_value.g);
    fragColor.b = mix(src_color.b, dst_color.b, mix_value.b);

    //  fragColor = vec4(mask_value, 1.0);
}