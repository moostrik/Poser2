// The 2D pressure's walls, as the 3D's (walls.glsl): beyond a closed wall the ghost is the cell's own pressure
// (Neumann), beyond an open one its negative, so the pressure is 0 on the wall's face (Dirichlet) on every level
uniform bool uClosed;

float wall_pressure(float p) {
    return uClosed ? p : -p;
}
