#version 460 core

uniform vec4 box_color;

out vec4 frag_color;

void main() {
    frag_color = box_color;
}
