#version 460 core

// Gaussian Blur Shader
// Ported from ofxFlowTools ftGaussianBlurShader.h
// Separable blur using binomial weights

uniform sampler2D tex0;
uniform float radius;
uniform int horizontal;
uniform vec2 texel_size;

in vec2 texCoord;
out vec4 fragColor;

// Binomial weights: 1, 8, 28, 56, 70, 56, 28, 8, 1
const float total = 252.0;  // Sum of weights

void main() {
    vec2 direction = horizontal == 1 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec2 offset_scale = direction * texel_size * radius;

    vec4 color = vec4(0.0);

    // Center sample (weight 70)
    color += (70.0 / total) * texture(tex0, texCoord);

    // Offset samples (4 steps on each side)
    color += (1.0 / total) * texture(tex0, texCoord - offset_scale * (4.0 / 4.0));
    color += (8.0 / total) * texture(tex0, texCoord - offset_scale * (3.0 / 4.0));
    color += (28.0 / total) * texture(tex0, texCoord - offset_scale * (2.0 / 4.0));
    color += (56.0 / total) * texture(tex0, texCoord - offset_scale * (1.0 / 4.0));

    color += (56.0 / total) * texture(tex0, texCoord + offset_scale * (1.0 / 4.0));
    color += (28.0 / total) * texture(tex0, texCoord + offset_scale * (2.0 / 4.0));
    color += (8.0 / total) * texture(tex0, texCoord + offset_scale * (3.0 / 4.0));
    color += (1.0 / total) * texture(tex0, texCoord + offset_scale * (4.0 / 4.0));

    fragColor = color;
}
