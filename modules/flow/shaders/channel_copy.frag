#version 460 core

// Channel Copy Shader
// Copies single-channel texture to specific RGBA channel while preserving others

uniform sampler2D dst;      // Existing RGBA texture (to preserve other channels)
uniform sampler2D src;      // Single-channel source (R32F)
uniform int channel;        // Target channel: 0=R, 1=G, 2=B, 3=A

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 existing = texture(dst, texCoord);
    float value = texture(src, texCoord).r;

    // Branchless channel selection using mix()
    // mask = vec4(1,0,0,0) for R, vec4(0,1,0,0) for G, etc.
    vec4 mask = vec4(equal(ivec4(0, 1, 2, 3), ivec4(channel)));

    // Replace only masked channel, preserve others
    fragColor = mix(existing, vec4(value), mask);
}
