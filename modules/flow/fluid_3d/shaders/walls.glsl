// The box's walls: one rule for every pass that reads beyond the box's edge, included where it reads (`#include
// "walls.glsl"`). Per axis a wall is closed or open (`WallMode`, uClosed 1 or 0):
//   closed  solid and free-slip, the textbook's (Bridson): nothing passes through, the water slides along
//   open    the water leaves: the pressure 0 on the wall's face, the velocity carried out unchanged
// A value beyond the wall is a ghost cell's, mirrored from the cell inside: the projection, the curl and the
// confinement read the same wall, so none of them sees one the others do not.
//
// Needs: uniform ivec3 uClosed;  (per axis, 1 closed)

// Whether `cell` lies beyond the box on `axis`
bool beyond(ivec3 cell, ivec3 size, int axis) {
    return cell[axis] < 0 || cell[axis] >= size[axis];
}

// The pressure beyond the wall on `axis`, the cell inside holding `p`: closed, the same (no gradient across, no
// push through the wall); open, its negative, so the pressure is 0 on the wall's face, at every level of the
// multigrid alike (a ghost of 0 would put that 0 half a cell further out on each level, a different place)
float wall_pressure(float p, int axis) {
    return uClosed[axis] == 1 ? p : -p;
}

// The velocity beyond the wall on `axis`, the cell inside moving `v`: closed, its component through the wall
// turned, so the face between them carries nothing, and along the wall the same, so the water slides with no
// shear (free-slip); open, the same (zero gradient), so it leaves as it came
vec3 wall_velocity(vec3 v, int axis) {
    if (uClosed[axis] == 0) return v;
    v[axis] = -v[axis];
    return v;
}
