#version 460 core

// Subtract pressure gradient from velocity to make divergence-free
// v_new = v_old - ∇p

in vec2 texCoord;
out vec2 fragColor;

uniform sampler2D uVelocity;   // Current velocity (RG32F)
uniform sampler2D uPressure;   // Pressure field (R32F)
uniform sampler2D uObstacle;   // Obstacle mask (R8, CLAMP_TO_BORDER=1)
uniform bool uHasObstacles;    // Skip obstacle checks when false

uniform vec2 uHalfRdxInv;  // 0.5 / the cell, widths, on x and y

#include "walls_2d.glsl"

void main() {
    vec2 st = texCoord;

    // Obstacle neighbor values (0 when no obstacles — BCs and clamping inert)
    float oT = 0.0, oB = 0.0, oR = 0.0, oL = 0.0;

    if (uHasObstacles) {
        float obstacle = texture(uObstacle, st).x;
        if (obstacle > 0.5) {
            fragColor = vec2(0.0);
            return;
        }

        oT = textureOffset(uObstacle, st, ivec2(0, 1)).r;
        oB = textureOffset(uObstacle, st, ivec2(0, -1)).r;
        oR = textureOffset(uObstacle, st, ivec2(1, 0)).r;
        oL = textureOffset(uObstacle, st, ivec2(-1, 0)).r;
    }

    // Sample pressure at neighbors
    float pT = textureOffset(uPressure, st, ivec2(0, 1)).x;
    float pB = textureOffset(uPressure, st, ivec2(0, -1)).x;
    float pR = textureOffset(uPressure, st, ivec2(1, 0)).x;
    float pL = textureOffset(uPressure, st, ivec2(-1, 0)).x;
    float pW = wall_pressure(texture(uPressure, st).x);
    // Beyond the field the wall's ghost (walls_2d.glsl)
    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(uPressure, 0);
    if (cell.y + 1 >= size.y) pT = pW;
    if (cell.y - 1 < 0)       pB = pW;
    if (cell.x + 1 >= size.x) pR = pW;
    if (cell.x - 1 < 0)       pL = pW;

    if (uHasObstacles) {
        float pC = texture(uPressure, st).x;

        // Neumann boundary conditions: zero gradient at obstacles
        pT = mix(pT, pC, oT);
        pB = mix(pB, pC, oB);
        pR = mix(pR, pC, oR);
        pL = mix(pL, pC, oL);
    }

    // Compute and subtract gradient (aspect-corrected)
    vec2 grad = vec2(uHalfRdxInv.x * (pR - pL), uHalfRdxInv.y * (pT - pB));
    vec2 vOld = texture(uVelocity, st).xy;
    vec2 vNew = vOld - grad;

    // No-penetration: zero velocity component pointing into obstacle neighbors
    if (oR > 0.5) vNew.x = min(vNew.x, 0.0);
    if (oL > 0.5) vNew.x = max(vNew.x, 0.0);
    if (oT > 0.5) vNew.y = min(vNew.y, 0.0);
    if (oB > 0.5) vNew.y = max(vNew.y, 0.0);

    fragColor = vNew;
}
