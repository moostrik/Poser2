// The velocity on the cells' faces (the MAC grid): x on the faces between left and right neighbors, y between
// upper and lower, z between front and back, included where a pass reads or writes them (`#include
// "faces.glsl"`). One RGBA16F volume a face longer on every axis (`FaceVelocity`): a texel's x, y and z are the
// faces on its cell's left, bottom and front, w unused. Along its own axis a component's faces run 0 to the cell
// count, the box's two walls included. Positions are in cells from the box's corner: x's face (i, j, k) sits at
// (i, j + 0.5, k + 0.5). A read is clamped to a component's own faces, so beyond the box it reads the face at the
// edge: the wall's own face across, its neighbor's along.
//
// Needs, declared before: uniform ivec3 uCells (the cell counts), uniform ivec3 uClosed and walls.glsl, uniform
// bool uHasObstacles, sampler3D uObstacle (R8 at the cells). `#define FACES_IN_PLACE` before it for a pass that
// reads and writes the same faces (`load_faces`).

layout(binding = 4) uniform sampler3D uFaces;

uniform vec3 uCell;           // a cell's sides in widths, x, y and z (`bind_faces`): the box's spacing, stretched in z

#ifdef FACES_IN_PLACE
layout(rgba16f, binding = 4) uniform image3D uOutFaces;
#else
layout(rgba16f, binding = 4) writeonly uniform image3D uOutFaces;
#endif

// The faces component `c` has along each axis: one more than the cells along its own
ivec3 face_count(int c) {
    ivec3 count = uCells;
    count[c] += 1;
    return count;
}

// Whether index `i` holds a face of component `c`
bool is_face(int c, ivec3 i) {
    return all(greaterThanEqual(i, ivec3(0))) && all(lessThan(i, face_count(c)));
}

// The position of component `c`'s face `i`, in cells
vec3 face_position(int c, ivec3 i) {
    vec3 p = vec3(i) + 0.5;
    p[c] -= 0.5;
    return p;
}

// Component `c` interpolated at `p` (cells) from its own faces. Beyond a closed wall it reads the face at the
// edge (the wall's own across, its neighbor's along: the free-slip mirror); beyond an open wall the water
// outside, at rest, 0, faded in over the last texel as a border of 0 is: so a trace from outside carries
// still water in, and the boundary never feeds itself
float face_at(int c, vec3 p) {
    vec3 t = p;
    t[c] += 0.5;                                        // a face's texel center is half a cell along its axis
    vec3 count = vec3(face_count(c));
    float outside = 1.0;
    for (int axis = 0; axis < 3; axis++) {
        if (uClosed[axis] == 0) {
            outside *= clamp(t[axis] + 0.5, 0.0, 1.0) * clamp(count[axis] + 0.5 - t[axis], 0.0, 1.0);
        }
    }
    t = clamp(t, vec3(0.5), count - 0.5);
    return outside * texture(uFaces, t / vec3(uCells + 1))[c];
}

// The velocity at `p` (cells), each component from its own faces (Bridson)
vec3 velocity_at(vec3 p) {
    return vec3(face_at(0, p), face_at(1, p), face_at(2, p));
}

// Component `c`'s face `i` as stored, clamped to its faces
float face_fetch(int c, ivec3 i) {
    return texelFetch(uFaces, clamp(i, ivec3(0), face_count(c) - 1), 0)[c];
}

// The cell `i`'s velocity, its faces averaged
vec3 cell_velocity(ivec3 i) {
    vec3 here = texelFetch(uFaces, i, 0).xyz;
    return 0.5 * (here + vec3(face_fetch(0, i + ivec3(1, 0, 0)), face_fetch(1, i + ivec3(0, 1, 0)),
                              face_fetch(2, i + ivec3(0, 0, 1))));
}

// The three components of index `i` written; a component the index holds no face of is never read
void store_faces(ivec3 i, vec3 v) {
    imageStore(uOutFaces, i, vec4(v, 0.0));
}

#ifdef FACES_IN_PLACE
vec3 load_faces(ivec3 i) {
    return imageLoad(uOutFaces, i).xyz;
}
#endif

// Whether component `c`'s face `i` lies on a closed wall
bool on_closed_wall(int c, ivec3 i) {
    return uClosed[c] == 1 && (i[c] == 0 || i[c] == uCells[c]);
}

// Whether cell `i` is an obstacle's (static: its faces' velocity 0); beyond the box none is
bool solid_cell(ivec3 i) {
    if (!uHasObstacles || any(lessThan(i, ivec3(0))) || any(greaterThanEqual(i, uCells))) return false;
    return texelFetch(uObstacle, i, 0).r > 0.5;
}

// Whether component `c`'s face `i` is solid: an obstacle's cell on either side
bool solid_face(int c, ivec3 i) {
    ivec3 before = i;
    before[c] -= 1;
    return solid_cell(i) || solid_cell(before);
}

// Whether component `c`'s face `i` is held at 0: on a closed wall or solid
bool held_face(int c, ivec3 i) {
    return on_closed_wall(c, i) || solid_face(c, i);
}
