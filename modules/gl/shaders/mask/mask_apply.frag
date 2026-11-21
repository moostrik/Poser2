#version 460 core

layout(binding = 0) uniform sampler2D   src;
layout(binding = 1) uniform sampler2D   mask;
uniform float       multiply;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 col_texel = texture(src, texCoord).rgb;
    float mask_texel = texture(mask, texCoord).r * multiply;
    fragColor = vec4(col_texel, mask_texel);
}