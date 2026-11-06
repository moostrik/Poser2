#version 460 core

// Cube Axes: each axis in its colormap color

in vec3 axis_color;
out vec4 fragColor;

void main() {
    fragColor = vec4(axis_color, 1.0);
}
