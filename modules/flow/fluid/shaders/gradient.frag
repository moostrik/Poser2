#version 460 core

// Subtract pressure gradient from velocity to make divergence-free
// v_new = v_old - ∇p

in vec2 texCoord;
out vec2 fragColor;

uniform sampler2D uVelocity;   // Current velocity (RG32F)
uniform sampler2D uPressure;   // Pressure field (R32F)
uniform sampler2D uObstacle;   // Obstacle mask (R8, CLAMP_TO_BORDER=1)

uniform vec2 uHalfRdxInv;  // (0.5/gridScale_x, 0.5/gridScale_y) for aspect correction

void main() {
    vec2 st = texCoord;

    float obstacle = texture(uObstacle, st).x;
    if (obstacle > 0.5) {
        fragColor = vec2(0.0);
        return;
    }

    // Sample pressure at neighbors
    float pT = textureOffset(uPressure, st, ivec2(0, 1)).x;
    float pB = textureOffset(uPressure, st, ivec2(0, -1)).x;
    float pR = textureOffset(uPressure, st, ivec2(1, 0)).x;
    float pL = textureOffset(uPressure, st, ivec2(-1, 0)).x;
    float pC = texture(uPressure, st).x;

    // Inline neighbor obstacle sampling
    float oT = textureOffset(uObstacle, st, ivec2(0, 1)).r;
    float oB = textureOffset(uObstacle, st, ivec2(0, -1)).r;
    float oR = textureOffset(uObstacle, st, ivec2(1, 0)).r;
    float oL = textureOffset(uObstacle, st, ivec2(-1, 0)).r;

    // Neumann boundary conditions: zero gradient at obstacles
    pT = mix(pT, pC, oT);
    pB = mix(pB, pC, oB);
    pR = mix(pR, pC, oR);
    pL = mix(pL, pC, oL);

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
