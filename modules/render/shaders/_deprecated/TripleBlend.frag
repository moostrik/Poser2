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

    float t1 = smoothstep(0.0, 1.0, blend1);
    float t2 = smoothstep(0.0, 1.0, blend2);

    float totalBlend = t1 + t2;
    vec4 blended_tex = totalBlend > 0.0 ? (texel1 * t1 + texel2 * t2) / totalBlend : vec4(0.0);

    float b0 = max(1.0 - blend1 - blend2, 0.0);
    fragColor = texel0 * b0 + blended_tex * (1.0 - b0);

    // float totalblend = blend1 + blend2
    // float b0 = max(1.0 - blend1 - blend2, 0.0);
    // float b1 = blend1;
    // float b2 = blend2;
    // fragColor = texel0 * b0 + texel1 * b1 + texel2 * b2;
}