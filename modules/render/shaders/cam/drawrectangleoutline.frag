#version 460 core

uniform vec4 color;
uniform float lineWidth;  // Line width in normalized coords

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Distance from edge (0 at edge, 0.5 at center)
    vec2 dist = abs(texCoord - 0.5);
    float maxDist = max(dist.x, dist.y);

    // Draw outline only
    float edge = 0.5 - lineWidth;
    if (maxDist < edge) {
        discard;
    }

    fragColor = color;
}
