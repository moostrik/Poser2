#version 460 core

precision highp float;

in vec2 texCoord;
out vec4 fragColor;

// Input mask textures (R16F single channel each)
uniform sampler2D uMask0;
uniform sampler2D uMask1;
uniform sampler2D uMask2;

// Per-mask colors and weights
uniform vec4 uColors[3];
uniform float uWeights[3];

void main() {
    // Sample masks (single channel R16F)
    float m0 = texture(uMask0, texCoord).r * uWeights[0];
    float m1 = texture(uMask1, texCoord).r * uWeights[1];
    float m2 = texture(uMask2, texCoord).r * uWeights[2];

    // Combined alpha from all masks
    float totalAlpha = clamp(m0 + m1 + m2, 0.0, 1.0);

    // Weighted color blend (non-premultiplied)
    // Each mask contributes its color proportionally
    vec3 color = vec3(0.0);
    if (totalAlpha > 0.001) {
        color = (m0 * uColors[0].rgb + m1 * uColors[1].rgb + m2 * uColors[2].rgb) / totalAlpha;
    }

    // Output non-premultiplied RGBA
    fragColor = vec4(color, totalAlpha);
}
