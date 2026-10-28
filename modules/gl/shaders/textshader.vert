#version 460 core

layout(location = 0) in vec2 position;   // screen pixels, one vertex per glyph-quad corner
layout(location = 1) in vec2 texcoord;   // font atlas UV

uniform vec2 screen_size;

out vec2 tex_coord;

void main() {
    // Convert pixel coordinates (top-left origin) to NDC
    vec2 ndc = (position / screen_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;

    gl_Position = vec4(ndc, 0.0, 1.0);
    tex_coord = texcoord;
}
