#version 460 core

// The player cut from depth: 1 where the metres lie from nearDepth to farDepth, falling off over
// `edge` metres beyond both ends; no depth (0) is outside. A single channel, red.

uniform sampler2D depth;
uniform float nearDepth;
uniform float farDepth;
uniform float edge;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float metres = texture(depth, texCoord).r;
    float soft = max(edge, 0.0001);
    float inside = metres > 0.0
        ? smoothstep(nearDepth - soft, nearDepth, metres) * (1.0 - smoothstep(farDepth, farDepth + soft, metres))
        : 0.0;
    fragColor = vec4(inside, 0.0, 0.0, 1.0);
}
