#version 460 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texcoord;

uniform vec4 rect;  // x, y, width, height (normalized 0..1, top-left origin, Y-down)

out vec2 texCoord;

void main() {
    // Transform quad vertices to rectangle position
    vec2 rectPos = position * 0.5 + 0.5;  // -1..1 to 0..1
    vec2 pos01 = rect.xy + rectPos * rect.zw;  // position in 0..1
    
    // Convert to NDC (-1..1, flip Y for top-left origin)
    vec2 ndc = pos01 * 2.0 - 1.0;
    ndc.y = -ndc.y;
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    texCoord = texcoord;
}
