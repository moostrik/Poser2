#version 460 core

// Velocity Impulse Region Speed Shader
// The speeds of the two trails, whose mip levels are a region's mean speeds: what a fall reads

uniform sampler2D tex0;  // Fast trail RGB
uniform sampler2D tex1;  // Slow trail RGB

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(length(texture(tex0, texCoord).xyz), length(texture(tex1, texCoord).xyz), 0.0, 1.0);
}
