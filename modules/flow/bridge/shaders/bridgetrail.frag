#version 460 core

// Bridge Trail Shader
// Ported from ofxFlowTools ftBridgeShader.h
// Blends previous trail with new input for temporal smoothing

uniform sampler2D tex0;  // Previous trail
uniform sampler2D tex1;  // New velocity input
uniform float weight;     // Trail weight (0=replace, 0.99=keep trail)

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.000001

void main() {
    vec2 prev_vel = texture(tex0, texCoord).xy;
    vec2 new_vel = texture(tex1, texCoord).xy;

    // Weighted blend: trail * weight + new
    vec2 vel = (prev_vel * weight) + new_vel;

    // Normalize and clamp magnitude to prevent accumulation
    float magnitude = min(length(vel), 1.0);
    vel = normalize(vel + TINY) * magnitude;

    fragColor = vec4(vel, 0.0, 1.0);
}
