#version 460 core

#include "position.glsl"

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform vec4 rect;  // x, y, width, height in 0..1, origin top-left

out vec2 texCoord;

void main() {
    gl_Position = clipPosition(rect.xy + topLeft(texcoord) * rect.zw);
    texCoord = texcoord;
}
