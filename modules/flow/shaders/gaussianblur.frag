#version 460 core

// Optimized Gaussian Blur using linear sampling
// Reduces texture fetches by using GPU's linear interpolation

uniform sampler2D tex0;
uniform float radius;
uniform int horizontal;
uniform vec2 resolution;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Calculate blur direction and step size
    vec2 direction = horizontal == 1 ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec2 texel_size = 1.0 / resolution;
    vec2 offset = direction * texel_size;

    // Clamp radius to reasonable range
    float r = clamp(radius, 0.0, 10.0);

    // Use linear sampling optimization
    // Each sample fetches 2 pixels and blends them (thanks to GL_LINEAR)
    vec4 color = vec4(0.0);
    float total_weight = 0.0;

    // Center sample
    float weight = 1.0;
    color += texture(tex0, texCoord) * weight;
    total_weight += weight;

    // Optimized sampling using linear interpolation
    // Weights chosen for good gaussian approximation
    const int samples = 5;
    float offsets[samples] = float[](1.0, 2.33, 3.67, 5.0, 6.33);
    float weights[samples] = float[](0.27, 0.22, 0.14, 0.08, 0.04);

    for (int i = 0; i < samples; i++) {
        float offset_dist = offsets[i] * r / 10.0;
        float w = weights[i];

        // Sample both directions
        color += texture(tex0, texCoord + offset * offset_dist) * w;
        color += texture(tex0, texCoord - offset * offset_dist) * w;
        total_weight += w * 2.0;
    }

    fragColor = color / total_weight;
}
