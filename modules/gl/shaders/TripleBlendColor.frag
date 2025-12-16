#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;

uniform sampler2D mask0;
uniform sampler2D mask1;
uniform sampler2D mask2;

uniform float blend0;
uniform float blend1;
uniform float blend2;

uniform vec4 color0;
uniform vec4 color1;
uniform vec4 color2;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 c0 = texture(tex0, texCoord);
    vec4 c1 = texture(tex1, texCoord);
    vec4 c2 = texture(tex2, texCoord);

    float m0 = texture(mask0, texCoord).r;
    float m1 = texture(mask1, texCoord).r;
    float m2 = texture(mask2, texCoord).r;

    vec4 f0 = c0 * m0;
    vec4 f1 = c1 * m1 * blend1 * blend0;
    vec4 f2 = c2 * m2 * blend2 * blend0;

    fragColor = f0 + f1 + f2;
}