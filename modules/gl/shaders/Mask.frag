#version 460 core

uniform sampler2D color;
uniform sampler2D mask;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 col_texel = texture(color, texCoord).rgb;
    float mask_texel = texture(mask, texCoord).r;
    fragColor = vec4(col_texel, mask_texel);
}