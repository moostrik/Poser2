#version 460 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform vec2 screenSize;
uniform vec4 rect;  // x, y, width, height in pixels

void main() {
    // Scale unit quad to box size and position
    vec2 pos = position * 0.5 + 0.5;  // Convert from [-1,1] to [0,1]
    vec2 pixelPos = pos * rect.zw + rect.xy;

    // Convert to NDC (normalized device coordinates)
    vec2 ndc = (pixelPos / screenSize) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y for screen coordinates (top-left origin)

    gl_Position = vec4(ndc, 0.0, 1.0);
}
