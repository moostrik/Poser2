#version 460 core

// Cube Axes: the box's three edges from its first corner (x 0, y 0, the front layer) along x, y and z,
// each a quad `thickness` pixels wide across the screen, made from the vertex's index, so the draw
// reads no vertex buffer

uniform mat4 view_projection;   // the box's space to clip space
uniform vec3 box;               // the box's size, its width 1
uniform vec2 resolution;        // the target's pixels
uniform float thickness;        // the lines' width (pixels)

const vec3 CORNER = vec3(-0.5, -0.5, 0.5);         // the front face is at +z
const vec3 ENDS[3] = vec3[](vec3(0.5, -0.5, 0.5), vec3(-0.5, 0.5, 0.5), vec3(-0.5, -0.5, -0.5));
const vec3 COLORS[3] = vec3[](vec3(1.0, 0.5, 0.5), vec3(0.5, 1.0, 0.5), vec3(0.5, 0.5, 1.0));     // the colormap's x, y, z at 1
const vec2 QUAD[6] = vec2[](vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(0.0, -1.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

out vec3 axis_color;

void main() {
    int axis = gl_VertexID / 6;
    vec2 corner = QUAD[gl_VertexID % 6];
    vec4 a = view_projection * vec4(CORNER * box, 1.0);
    vec4 b = view_projection * vec4(ENDS[axis] * box, 1.0);
    vec2 along = (b.xy / b.w - a.xy / a.w) * resolution;
    vec2 normal = length(along) > 1e-6 ? normalize(vec2(-along.y, along.x)) : vec2(0.0, 1.0);
    vec4 p = mix(a, b, corner.x);
    p.xy += normal * corner.y * thickness / resolution * p.w;      // half the width each side, in clip space
    gl_Position = p;
    axis_color = COLORS[axis];
}
