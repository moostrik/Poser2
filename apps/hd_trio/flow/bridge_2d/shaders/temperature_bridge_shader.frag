#version 460 core

layout(binding = 0) uniform sampler2D uColor;      // Color input (RGB: R=warm, G=neutral, B=cold)
layout(binding = 1) uniform sampler2D uMask;       // Velocity magnitude mask (R32F)
uniform float uScale;          // Output multiplier

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 color = texture(uColor, texCoord).rgb;
    float mask = texture(uMask, texCoord).r;

    float warm = color.r;      // Red = heat
    float cold = color.b;      // Blue = cold
    float neutral = color.g;   // Green = dampening

    // Temperature = (warm - cold) dampened by green
    float temp = (warm - cold) * (1.0 - neutral * 0.5);
    temp *= mask * uScale;     // Apply velocity mask and scale

    fragColor = vec4(temp, 0.0, 0.0, 0.0);
}
