#version 460 core

// Depth Velocity Shader
// Velocity toward the camera from two smoothed depths, 0 without depth at both and at a jump.
// The numpy reference is kinetics.depth_speed

uniform sampler2D tex0;  // Smoothed depth R
uniform sampler2D tex1;  // Previous smoothed depth R
uniform float dt;        // Seconds between them
uniform float jump;      // More change than this is an edge (m)
uniform float scale;     // Converts m/s to simulation velocity

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float depth = texture(tex0, texCoord).r;
    float previous = texture(tex1, texCoord).r;
    float change = depth - previous;
    bool moving = depth > 0.0 && previous > 0.0 && abs(change) <= jump;
    fragColor = vec4(moving ? -change / dt * scale : 0.0, 0.0, 0.0, 1.0);
}
