#version 460 core

#include "position.glsl"

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

// Where the skeleton is drawn: x, y, width, height in 0..1 of the target, origin top-left. The whole target by
// default, so a layer that draws into its own FBO needs no rect.
uniform vec4 rect = vec4(0.0, 0.0, 1.0, 1.0);

out vec2 texCoord;

void main() {
    gl_Position = clipPosition(rect.xy + topLeft(texcoord) * rect.zw);
    texCoord = texcoord;
}
