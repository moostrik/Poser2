#version 460 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform vec2 screen_size;
uniform vec4 glyph_rect;  // x, y, width, height in pixels
uniform vec4 uv_rect;     // u0, v0, u1, v1

out vec2 tex_coord;

void main() {
    // Scale unit quad to glyph size and position
    vec2 pos = position * 0.5 + 0.5;  // Convert from [-1,1] to [0,1]
    vec2 pixel_pos = pos * glyph_rect.zw + glyph_rect.xy;

    // Convert to NDC (normalized device coordinates)
    vec2 ndc = (pixel_pos / screen_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y for screen coordinates (top-left origin)

    gl_Position = vec4(ndc, 0.0, 1.0);

    // Interpolate UV coordinates
    tex_coord = mix(uv_rect.xy, uv_rect.zw, pos);
}
