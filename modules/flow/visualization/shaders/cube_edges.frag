#version 460 core

// Cube Edges: a grey line, for the box's shape

uniform vec4 color;
out vec4 fragColor;

void main() {
    fragColor = color;
}
