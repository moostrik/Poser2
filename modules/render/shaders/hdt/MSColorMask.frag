#version 460 core

precision highp float;

in vec2 texCoord;
out vec4 fragColor;

// All 3 slots: pre-styled RGBA (already tinted)
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform float uWeights[3];

void main() {
    vec4 t0 = texture(uTex0, texCoord);
    vec4 t1 = texture(uTex1, texCoord);
    vec4 t2 = texture(uTex2, texCoord);

    float a0 = t0.a * uWeights[0];
    float a1 = t1.a * uWeights[1];
    float a2 = t2.a * uWeights[2];

    float totalAlpha = clamp(a0 + a1 + a2, 0.0, 1.0);
    vec3 premult = t0.rgb * a0 + t1.rgb * a1 + t2.rgb * a2;
    vec3 color = (totalAlpha > 0.001) ? premult / totalAlpha : vec3(0.0);

    fragColor = vec4(color, totalAlpha);
}
