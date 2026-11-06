#version 460 core

// Cube Edges: the box's twelve edges as lines, their corners made from the vertex's index, so the draw
// reads no vertex buffer

uniform mat4 view_projection;   // the box's space to clip space
uniform vec3 box;               // the box's size, its width 1

const ivec2 EDGES[12] = ivec2[](ivec2(0, 1), ivec2(1, 3), ivec2(3, 2), ivec2(2, 0),     // the back face
                                ivec2(4, 5), ivec2(5, 7), ivec2(7, 6), ivec2(6, 4),     // the front face
                                ivec2(0, 4), ivec2(1, 5), ivec2(2, 6), ivec2(3, 7));    // front to back

void main() {
    ivec2 edge = EDGES[gl_VertexID / 2];
    int corner = (gl_VertexID % 2 == 0) ? edge.x : edge.y;
    vec3 p = vec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1) - 0.5;     // bit 2 set: the front
    gl_Position = view_projection * vec4(p * box, 1.0);
}
