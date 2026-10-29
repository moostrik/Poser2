#version 460 core

uniform vec4 color;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = color;
}
