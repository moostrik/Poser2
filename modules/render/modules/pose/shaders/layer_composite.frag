#version 460 core

precision highp float;

in vec2 texCoord;
out vec4 fragColor;

// All 3 slots: pre-styled RGBA (already tinted)
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform float uWeights[3];
uniform float uLayered;  // 0 = additive, 1 = own in front

void main() {
    vec4 t0 = texture(uTex0, texCoord);
    vec4 t1 = texture(uTex1, texCoord);
    vec4 t2 = texture(uTex2, texCoord);

    vec3 c0 = t0.rgb * t0.a * uWeights[0];
    vec3 c1 = t1.rgb * t1.a * uWeights[1];
    vec3 c2 = t2.rgb * t2.a * uWeights[2];

    float a0 = t0.a * uWeights[0];
    float a1 = t1.a * uWeights[1];
    float a2 = t2.a * uWeights[2];

    // Additive mode
    vec3 addColor = c0 + c1 + c2;
    float addAlpha = clamp(a0 + a1 + a2, 0.0, 1.0);

    // Layered mode: own over additive others
    vec3 bg = c1 + c2;
    vec3 layerColor = c0 + bg * (1.0 - a0);
    float layerAlpha = clamp(a0 + (a1 + a2) * (1.0 - a0), 0.0, 1.0);

    // Mix based on uLayered
    vec3 color = mix(addColor, layerColor, uLayered);
    float alpha = mix(addAlpha, layerAlpha, uLayered);

    fragColor = vec4(color, alpha);
}
