#version 460 core

precision highp float;

in vec2 texCoord;
out vec4 fragColor;

// All 3 layers: pre-styled RGBA (already tinted)
uniform sampler2D layers[3];
uniform float weights[3];
uniform float layered;      // 0 = additive, 1 = the first in front

void main() {
    vec4 t0 = texture(layers[0], texCoord);
    vec4 t1 = texture(layers[1], texCoord);
    vec4 t2 = texture(layers[2], texCoord);

    vec3 c0 = t0.rgb * t0.a * weights[0];
    vec3 c1 = t1.rgb * t1.a * weights[1];
    vec3 c2 = t2.rgb * t2.a * weights[2];

    float a0 = t0.a * weights[0];
    float a1 = t1.a * weights[1];
    float a2 = t2.a * weights[2];

    // Additive mode
    vec3 addColor = c0 + c1 + c2;
    float addAlpha = clamp(a0 + a1 + a2, 0.0, 1.0);

    // Layered mode: the first over the others added
    vec3 bg = c1 + c2;
    vec3 layerColor = c0 + bg * (1.0 - a0);
    float layerAlpha = clamp(a0 + (a1 + a2) * (1.0 - a0), 0.0, 1.0);

    fragColor = vec4(mix(addColor, layerColor, layered), mix(addAlpha, layerAlpha, layered));
}
