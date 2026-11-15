#version 460 core

// The quad placed at `dstRect` in the target and textured from `srcRect`; both x, y, width, height in 0..1, origin
// top-left. A negative width or height of `srcRect` draws it mirrored on that axis.

#include "position.glsl"

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform vec4 dstRect;
uniform vec4 srcRect;

out vec2 texCoord;

void main() {
    vec2 corner = topLeft(texcoord);
    gl_Position = clipPosition(dstRect.xy + corner * dstRect.zw);
    texCoord = topLeft(srcRect.xy + corner * srcRect.zw);
}
