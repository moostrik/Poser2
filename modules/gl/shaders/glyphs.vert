#version 460 core

#include "position.glsl"

layout(location = 0) in vec2 position;   // in 0..1 of the target, origin top-left, one vertex per glyph-quad corner
layout(location = 1) in vec2 texcoord;   // font atlas UV

out vec2 texCoord;

void main() {
    gl_Position = clipPosition(position);
    texCoord = texcoord;
}
