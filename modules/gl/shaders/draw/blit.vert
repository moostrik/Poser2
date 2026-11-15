#version 460 core

// The quad placed at `dstRect` in the target and textured from the four corners `srcQuad` (top left, top right,
// bottom right, bottom left); all in 0..1, origin top-left. Corners that turn or mirror the rectangle draw it turned
// or mirrored.

#include "position.glsl"

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform vec4 dstRect;
uniform vec2 srcQuad[4];

out vec2 texCoord;

void main() {
    vec2 corner = topLeft(texcoord);
    gl_Position = clipPosition(dstRect.xy + corner * dstRect.zw);
    vec2 top = mix(srcQuad[0], srcQuad[1], corner.x);
    vec2 bottom = mix(srcQuad[3], srcQuad[2], corner.x);
    texCoord = topLeft(mix(top, bottom, corner.y));
}
