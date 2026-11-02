#version 460 core

// Range flow's velocity toward the camera (m/s, its blue channel) as the depth velocity's output, scaled.

uniform sampler2D motion;   // range flow: displacement (RG), toward the camera m/s (B)
uniform float scale;        // Converts m/s to simulation velocity

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(texture(motion, texCoord).b * scale, 0.0, 0.0, 1.0);
}
