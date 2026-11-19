// The 2D fluid's walls, as the 3D's (walls.glsl): one rule for every pass that reads beyond the field's edge. A wall
// is closed or open (`WallMode`, uClosed):
//   closed  solid: nothing passes through; the water slides along freely (free-slip), or slowed over `slip_time`
//           (Navier slip: uWallKeep, the share of the velocity along the wall kept a step)
//   open    the water leaves: the pressure 0 on the wall's face, the velocity carried out unchanged
// A value beyond the wall is a ghost cell's, mirrored from the cell inside: the projection, the curl and the
// viscosity read the same wall, so none of them sees one the others do not.
uniform bool uClosed;
uniform float uWallKeep = 1.0;  // the share of the velocity along a closed wall kept a step; 1 free-slip

// The pressure beyond the wall, the cell inside holding `p`: closed, the same (Neumann); open, its negative, so the
// pressure is 0 on the wall's face (Dirichlet) on every level of the multigrid alike
float wall_pressure(float p) {
    return uClosed ? p : -p;
}

// The velocity beyond the wall across `axis` (0 x, 1 y), the cell inside moving `v`: closed, its component through
// the wall turned, so the wall's face carries nothing, and along the wall (2 keep - 1) of it, so the face moves with
// the share the wall keeps; open, the same (zero gradient), so it leaves as it came
vec2 wall_velocity(vec2 v, int axis) {
    if (!uClosed) return v;
    float normal = -v[axis];
    v *= 2.0 * uWallKeep - 1.0;
    v[axis] = normal;
    return v;
}

// The velocity of a cell against a closed wall, braked along it by uWallKeep (the slip); `cell` of `size`
vec2 slip(vec2 v, ivec2 cell, ivec2 size) {
    if (!uClosed || uWallKeep >= 1.0) return v;
    if (cell.x == 0 || cell.x == size.x - 1) v.y *= uWallKeep;
    if (cell.y == 0 || cell.y == size.y - 1) v.x *= uWallKeep;
    return v;
}
