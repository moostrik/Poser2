#version 460 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;
out vec2 texCoord;
out vec2 clipPos;

void main() {
    texCoord = texcoord;
    clipPos = position;
    gl_Position = vec4(position, 0.0, 1.0);
}