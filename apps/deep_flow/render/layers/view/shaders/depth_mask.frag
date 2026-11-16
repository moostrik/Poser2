#version 460 core

// The player cut from depth: 1 where the metres lie from nearDepth to farDepth; no depth (0) is
// outside. A single channel, red.

uniform sampler2D depth;
uniform float nearDepth;
uniform float farDepth;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float metres = texture(depth, texCoord).r;
    float inside = metres > 0.0 && metres >= nearDepth && metres <= farDepth ? 1.0 : 0.0;
    fragColor = vec4(inside, 0.0, 0.0, 1.0);
}
