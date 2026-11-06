#version 460 core

// Glyph Arrows: a velocity volume as 3D arrows on a grid, one instance per grid point, the arrow a
// small mesh made from the vertex's index (a shaft and a head of 8 sides, 96 vertices), so the draw
// reads no vertex buffer. An arrow points along the velocity in the box (x right, y up, z toward the
// front), reaches one cell at a speed of 1 and rolls off toward two; colored by its direction

uniform sampler3D tex0;         // The velocity (xyz)
uniform mat4 view_projection;   // the box's space to clip space
uniform vec3 box;               // the box's size, its width 1
uniform ivec3 grid;             // the grid points across, down and deep
uniform float scale;            // Multiplier on the velocity

out vec3 normal;
out vec3 color;

const int SIDES = 8;
const float SHAFT_RADIUS = 0.07;    // shares of the arrow's cell
const float HEAD_RADIUS = 0.18;
const float SHAFT = 0.62;           // the shaft's share of the arrow's length
const float SIZE_LIMIT = 2.0;       // the speed the length rolls off toward
const float PI = 3.14159265;

// A point of the mesh in the arrow's frame: along +y from 0 to `len`, its normal
void mesh(int id, float len, float radius_scale, out vec3 p, out vec3 n) {
    int part = id / (SIDES * 6);                    // 0 the shaft's sides, 1 the head's cone and its base
    int local = id % (SIDES * 6);
    int side = local / 6;
    int corner = local % 6;
    float a0 = 2.0 * PI * float(side) / float(SIDES);
    float a1 = 2.0 * PI * float(side + 1) / float(SIDES);
    float shaft_end = len * SHAFT;
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
    if (speed < 1e-4) {                             // no arrow: every vertex at one point draws nothing
        gl_Position = view_projection * vec4(center, 1.0);
        normal = vec3(0.0, 0.0, 1.0);
        color = vec3(0.0);
        return;
    }
    vec3 dir = velocity / speed;
    float len = cell_size * SIZE_LIMIT * tanh(speed / SIZE_LIMIT);

    vec3 p, n;
    mesh(gl_VertexID, len, cell_size, p, n);

    vec3 helper = abs(dir.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 side = normalize(cross(helper, dir));
    vec3 other = cross(dir, side);
    mat3 frame = mat3(side, dir, other);            // the arrow's +y along the velocity
    normal = frame * n;
    color = 0.5 + 0.5 * dir;
    gl_Position = view_projection * vec4(center + frame * p, 1.0);
}
