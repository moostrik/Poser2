#version 460 core

// Glyph Arrows: a velocity volume as 3D arrows on a grid, one instance per grid point, the arrow a
// small mesh made from the vertex's index (a shaft and a head of 8 sides, 96 vertices), so the draw
// reads no vertex buffer. An arrow points along the velocity in the box (x right, y up, z toward the
// front), reaches one cell at a speed of 1 and rolls off toward two, and thickens with its length up
// to full at one cell; below a tenth of a cell long (at least 2 px) it is a dot. Colored by its direction

uniform sampler3D tex0;         // The velocity (xyz)
uniform mat4 view_projection;   // the box's space to clip space
uniform vec3 box;               // the box's size, its width 1
uniform ivec3 grid;             // the grid points across, down and deep
uniform float scale;            // Multiplier on the velocity
uniform float pixels_per_unit;  // pixels a box unit spans at a clip w of 1
uniform bool push;              // An acceleration: the cone alone, a push, no shaft

out vec3 normal;
out vec3 color;

const int SIDES = 8;
const float SHAFT_RADIUS = 0.07;    // shares of the arrow's cell, at full thickness
const float HEAD_RADIUS = 0.18;
const float SHAFT = 0.62;           // the shaft's share of the arrow's length
const float SIZE_LIMIT = 2.0;       // the speed the length rolls off toward
const float DOT = 0.1;              // the dot's diameter as a share of the cell; below it in length, a dot
const float DOT_PIXELS = 2.0;       // the dot's least diameter on screen
const int DOT_SIDES = 16;           // the dot: a double cone of 16 sides, 96 vertices as the arrow
const float PI = 3.14159265;

// A point of the dot, a double cone around the origin of radius `r`, and its normal
void dot_mesh(int id, float r, out vec3 p, out vec3 n) {
    int triangle = id / 3;
    int corner = id % 3;
    float apex = triangle < DOT_SIDES ? 1.0 : -1.0;       // the upper cone, then the lower
    int side = triangle % DOT_SIDES;
    float a = 2.0 * PI * float(corner == 1 ? side + 1 : side) / float(DOT_SIDES);
    p = corner == 2 ? vec3(0.0, apex * r, 0.0) : vec3(cos(a) * r, 0.0, sin(a) * r);
    n = normalize(p);
}

// A point of the mesh in the arrow's frame: along +y from 0 to `len`, its normal
void mesh(int id, float len, float radius_scale, float shaft, out vec3 p, out vec3 n) {
    int part = id / (SIDES * 6);                    // 0 the shaft's sides, 1 the head's cone and its base
    int local = id % (SIDES * 6);
    int side = local / 6;
    int corner = local % 6;
    float a0 = 2.0 * PI * float(side) / float(SIDES);
    float a1 = 2.0 * PI * float(side + 1) / float(SIDES);
    float shaft_end = len * shaft;          // 0 for a push: the shaft's sides collapse, the cone is all
    if (part == 0) {                                // two triangles of the shaft's side
        const int SEGMENT[6] = int[](0, 1, 2, 1, 3, 2);
        int c = SEGMENT[corner];
        float a = (c == 0 || c == 2) ? a0 : a1;
        float y = (c < 2) ? 0.0 : shaft_end;
        vec2 ring = vec2(cos(a), sin(a));
        p = vec3(ring.x * SHAFT_RADIUS * radius_scale, y, ring.y * SHAFT_RADIUS * radius_scale);
        n = vec3(ring.x, 0.0, ring.y);
    } else if (corner < 3) {                        // the cone, from the base's rim to the tip
        float a = corner == 0 ? a0 : (corner == 1 ? a1 : 0.5 * (a0 + a1));
        vec2 ring = vec2(cos(a), sin(a));
        float tilt = HEAD_RADIUS * radius_scale / max(len - shaft_end, 1e-4);
        n = normalize(vec3(ring.x, tilt, ring.y));
        p = corner == 2 ? vec3(0.0, len, 0.0) : vec3(ring.x * HEAD_RADIUS * radius_scale, shaft_end, ring.y * HEAD_RADIUS * radius_scale);
    } else {                                        // the cone's base, facing back
        float a = corner == 3 ? a1 : (corner == 4 ? a0 : 0.0);
        vec2 ring = vec2(cos(a), sin(a));
        p = corner == 5 ? vec3(0.0, shaft_end, 0.0) : vec3(ring.x * HEAD_RADIUS * radius_scale, shaft_end, ring.y * HEAD_RADIUS * radius_scale);
        n = vec3(0.0, -1.0, 0.0);
    }
}

void main() {
    int id = gl_InstanceID;
    ivec3 cell = ivec3(id % grid.x, (id / grid.x) % grid.y, id / (grid.x * grid.y));
    vec3 uvw = (vec3(cell) + 0.5) / vec3(grid);
    vec3 center = vec3((uvw.xy - 0.5) * box.xy, box.z * (0.5 - uvw.z));

    vec3 velocity = textureLod(tex0, uvw, 0.0).xyz * scale;
    float speed = length(velocity);
    float cell_size = box.x / float(grid.x);
    if (speed <= 0.0) {                             // no arrow: every vertex at one point draws nothing
        gl_Position = view_projection * vec4(center, 1.0);
        normal = vec3(0.0, 0.0, 1.0);
        color = vec3(0.0);
        return;
    }
    vec3 dir = velocity / speed;
    color = 0.5 + 0.5 * dir;
    float len = cell_size * SIZE_LIMIT * tanh(speed / SIZE_LIMIT);
    float w = (view_projection * vec4(center, 1.0)).w;
    float dot_size = max(DOT * cell_size, DOT_PIXELS * w / max(pixels_per_unit, 1e-6));

    vec3 p, n;
    if (len < dot_size) {                           // a tiny force: the dot, in its direction's color
        dot_mesh(gl_VertexID, 0.5 * dot_size, p, n);
        normal = n;
        gl_Position = view_projection * vec4(center + p, 1.0);
        return;
    }
    // the arrow thickens with its length up to full at one cell, its head never narrower than the dot
    float thickness = cell_size * max(min(len / cell_size, 1.0), 0.5 * dot_size / (HEAD_RADIUS * cell_size));
    mesh(gl_VertexID, len, thickness, push ? 0.0 : SHAFT, p, n);

    vec3 helper = abs(dir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 side = normalize(cross(helper, dir));
    vec3 other = cross(dir, side);
    mat3 frame = mat3(side, dir, other);            // the arrow's +y along the velocity
    normal = frame * n;
    gl_Position = view_projection * vec4(center + frame * p, 1.0);
}
