#version 460 core

// Channel Add Shader
// Adds single-channel texture to specific RGBA channel while preserving others

uniform sampler2D dst;      // Existing RGBA texture (to preserve and add to)
uniform sampler2D src;      // Single-channel source (R32F)
uniform int channel;        // Target channel: 0=R, 1=G, 2=B, 3=A
uniform float strength;     // Strength multiplier for the addition

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 existing = texture(dst, texCoord);
    float value = texture(src, texCoord).r * strength;

    // Branchless channel selection using mix()
    // mask = vec4(1,0,0,0) for R, vec4(0,1,0,0) for G, etc.
    vec4 mask = vec4(equal(ivec4(0, 1, 2, 3), ivec4(channel)));

    // Add to only the masked channel, preserve others
    fragColor = existing + vec4(value) * mask;
}
