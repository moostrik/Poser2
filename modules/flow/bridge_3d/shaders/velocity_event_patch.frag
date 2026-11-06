#version 460 core

// Velocity Event Patch Shader
// The x/y velocities of the two trails, whose mip levels are the mean velocity of a patch: the start
// gate's. Jitter in every direction cancels in a mean velocity, where it adds up in a mean speed

uniform sampler2D tex0;  // Fast trail RGB
uniform sampler2D tex1;  // Slow trail RGB

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(texture(tex0, texCoord).xy, texture(tex1, texCoord).xy);
}
