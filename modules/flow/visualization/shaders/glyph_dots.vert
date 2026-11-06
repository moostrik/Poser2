#version 460 core

// Glyph Dots: a scalar volume as 3D dots on a grid, one point per grid point made from the vertex's
// index, so the draw reads no vertex buffer; its size on screen the value's (half a cell at 1, small
// values small), its color the colormap's

uniform sampler3D tex0;         // The volume
uniform mat4 view_projection;   // the box's space to clip space
uniform vec3 box;               // the box's size, its width 1
uniform ivec3 grid;             // the grid points across, down and deep
uniform float scale;            // Multiplier on the values
uniform int channel;            // 0 to 3; 4 for all: x's size and xyz's colors
uniform float pixels_per_unit;  // the viewport's height over the projection's, times its focal length: pixels per box unit at depth 1

out vec3 color;
out float amount;

const vec3 AXES[4] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(1.0));

void main() {
    int id = gl_VertexID;
    ivec3 cell = ivec3(id % grid.x, (id / grid.x) % grid.y, id / (grid.x * grid.y));
    vec3 uvw = (vec3(cell) + 0.5) / vec3(grid);
    vec3 center = vec3((uvw.xy - 0.5) * box.xy, box.z * (0.5 - uvw.z));

    vec4 value = textureLod(tex0, uvw, 0.0) * scale;
    amount = channel == 4 ? value.x : value[channel];
    color = clamp(0.5 + 0.5 * (channel == 4 ? value.xyz : AXES[channel] * amount), 0.0, 1.0);

    gl_Position = view_projection * vec4(center, 1.0);
    float cell_size = box.x / float(grid.x);
    gl_PointSize = max(pixels_per_unit * cell_size * tanh(abs(amount)) / max(gl_Position.w, 1e-4), 0.0);
}
