#version 460 core

// Velocity Bandpass Region Speed 2D Shader
// The speeds of the two trails, whose mip levels are a region's mean speeds: what a fall reads

layout(binding = 0) uniform sampler2D tex0;  // Fast trail RG
layout(binding = 1) uniform sampler2D tex1;  // Slow trail RG

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(length(texture(tex0, texCoord).xy), length(texture(tex1, texCoord).xy), 0.0, 0.0);
}
