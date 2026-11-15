#version 460 core

uniform vec4 color;
uniform vec2 lineWidth;  // Line width in normalized coords (x, y)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Distance from edge (0 at edge, 0.5 at center)
    vec2 dist = abs(texCoord - 0.5);
    vec2 edge = vec2(0.5) - lineWidth;

    // Draw outline only - discard if inside the border
    if (dist.x < edge.x && dist.y < edge.y) {
        discard;
    }

    fragColor = color;
}
