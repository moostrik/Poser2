#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;

uniform float blend1;
uniform float blend2;

uniform vec4 color0;
uniform vec4 color1;
uniform vec4 color2;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float texel0 = texture(tex0, texCoord).r;
    float texel1 = texture(tex1, texCoord).r;
    float texel2 = texture(tex2, texCoord).r;

    vec4 c0 = color0 * texel0;
    vec4 c1 = color1 * texel1;
    vec4 c2 = color2 * texel2;

    c0.a*= 1.0;
    c1.a*= blend1 * texel0;
    c2.a*= blend2 * texel0;

    fragColor = c0 + c1 + c2;

    // float t1 = smoothstep(0.0, 1.0, blend1);
    // float t2 = smoothstep(0.0, 1.0, blend2);

    // float totalBlend = t1 + t2;
    // vec4 blended_tex = totalBlend > 0.0 ? (texel1 * t1 + texel2 * t2) / totalBlend : vec4(0.0);

    // float b0 = max(1.0 - blend1 - blend2, 0.0);
    // fragColor = texel0 * b0 + blended_tex * (1.0 - b0);

    // float totalblend = blend1 + blend2
    // float b0 = max(1.0 - blend1 - blend2, 0.0);
    // float b1 = blend1;
    // float b2 = blend2;
    // fragColor = texel0 * b0 + texel1 * b1 + texel2 * b2;
}