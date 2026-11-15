#version 460 core

layout(location = 0) in vec2 position;   // screen pixels, one vertex per glyph-quad corner
layout(location = 1) in vec2 texcoord;   // font atlas UV

uniform vec2 screenSize;

out vec2 texCoord;

void main() {
    // Convert pixel coordinates (top-left origin) to NDC
    vec2 ndc = (position / screenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
    texCoord = texcoord;
}
