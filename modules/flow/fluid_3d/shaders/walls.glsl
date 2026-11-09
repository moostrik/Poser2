// The box's walls: one rule for every pass that reads beyond the box's edge, included where it reads (`#include
// "walls.glsl"`). Per axis a wall is closed or open (`WallMode`, uClosed 1 or 0):
//   closed  solid: nothing passes through; the water slides along freely (free-slip, the textbook's,
//           Bridson), or slowed over `slip_time` (Navier slip, the layer along the wall a coarse grid
//           cannot resolve, modelled: uWallKeep, the share of the velocity along the wall kept a step)
//   open    the water leaves: the pressure 0 on the wall's face, the velocity carried out unchanged
// A value beyond the wall is a ghost cell's, mirrored from the cell inside: the projection, the curl and the
// confinement read the same wall, so none of them sees one the others do not.
//
// Needs: uniform ivec3 uClosed;  (per axis, 1 closed)

uniform float uWallKeep = 1.0;  // the share of the velocity along a closed wall kept a step; 1 free-slip

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
// turned, so the face between them carries nothing, and along the wall (2 keep - 1) of it, so the face moves
// with the share the wall keeps: all of it free-slip (no shear), none with the water held; open, the same
// (zero gradient), so it leaves as it came
vec3 wall_velocity(vec3 v, int axis) {
    if (uClosed[axis] == 0) return v;
    float normal = -v[axis];
    v *= 2.0 * uWallKeep - 1.0;
    v[axis] = normal;
    return v;
}

