#version 460 core

// Subtract pressure gradient from velocity to make divergence-free
// v_new = v_old - ∇p

in vec2 texCoord;
out vec2 fragColor;

uniform sampler2D uVelocity;        // Current velocity (RG32F)
uniform sampler2D uPressure;        // Pressure field (R32F)
uniform sampler2D uObstacle;        // Obstacle mask (R8/R32F)
uniform sampler2D uObstacleOffset;  // Neighbor obstacle info (RGBA8)

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

    // Neumann boundary conditions (zero gradient at obstacles)
    // Use pC if neighbor is obstacle
    vec4 oN = texture(uObstacleOffset, st);
    pT = mix(pT, pC, oN.x);
    pB = mix(pB, pC, oN.y);
    pR = mix(pR, pC, oN.z);
    pL = mix(pL, pC, oN.w);

    // Compute and subtract gradient (aspect-corrected)
    vec2 grad = vec2(uHalfRdxInv.x * (pR - pL), uHalfRdxInv.y * (pT - pB));
    vec2 vOld = texture(uVelocity, st).xy;

    fragColor = vOld - grad;
}
