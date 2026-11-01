#version 460 core

// Speed 3D Shader
// The speed of (x, y, z), as a length the density and temperature bridges read like a 2D velocity

uniform sampler2D tex0;  // Velocity RG
uniform sampler2D tex1;  // Velocity toward the camera R

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 velocity = texture(tex0, texCoord).xy;
    float depth_velocity = texture(tex1, texCoord).r;
    fragColor = vec4(length(vec3(velocity, depth_velocity)), 0.0, 0.0, 1.0);
}
