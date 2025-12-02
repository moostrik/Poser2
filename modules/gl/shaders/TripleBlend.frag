#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform float blend1;
uniform float blend2;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 texel0 = texture(tex0, texCoord);
    vec4 texel1 = texture(tex1, texCoord);
    vec4 texel2 = texture(tex2, texCoord);
    float b0 = 1.0 - blend1 - blend2;
    float b1 = blend1;
    float b2 = blend2;
    fragColor = texel0 * b0 + texel1 * b1 + texel2 * b2;
}