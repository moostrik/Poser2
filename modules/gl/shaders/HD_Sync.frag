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

    float w0 = 1.0 - blend1 - blend2;
    float w1 = blend1;
    float w2 = blend2;

    vec4 color = texel0 + texel1 * blend1 + texel2 * blend2 * 2.0;
    float magnitude = length(color.rgb);
    float t = 1.3;
    if (magnitude > t) {
        color.rgb = normalize(color.rgb) * t;
    }
    fragColor = color * 2.5;
}