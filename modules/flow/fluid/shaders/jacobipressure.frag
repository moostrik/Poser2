#version 460 core

// Jacobi iterative solver for Poisson pressure equation
// Solves: ∇²p = -∇·v

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D uSource;          // Previous pressure estimate (R32F)
uniform sampler2D uDivergence;      // Velocity divergence (R32F)
uniform sampler2D uObstacle;        // Obstacle mask (R8/R32F)
uniform sampler2D uObstacleOffset;  // Neighbor obstacle info (RGBA8)

uniform float uAlpha;  // -(gridScale²)
uniform float uBeta;   // 0.25 (= 1/4 for 4 neighbors)

void main() {
    vec2 st = vTexCoord;

    float obstacle = texture(uObstacle, st).x;
    if (obstacle > 0.5) {
        fragColor = vec4(0.0);
        return;
    }

    // Sample divergence (RHS of equation)
    vec4 bC = texture(uDivergence, st);

    // Sample pressure at neighbors
    vec4 xT = textureOffset(uSource, st, ivec2(0, 1));
    vec4 xB = textureOffset(uSource, st, ivec2(0, -1));
    vec4 xR = textureOffset(uSource, st, ivec2(1, 0));
    vec4 xL = textureOffset(uSource, st, ivec2(-1, 0));
    vec4 xC = texture(uSource, st);

    // Neumann boundary conditions (zero gradient at obstacles)
    // Use xC if neighbor is obstacle
    vec4 oN = texture(uObstacleOffset, st);
    xT = mix(xT, xC, oN.x);
    xB = mix(xB, xC, oN.y);
    xR = mix(xR, xC, oN.z);
    xL = mix(xL, xC, oN.w);

    // Jacobi iteration: x^(k+1) = (xL + xR + xB + xT + α*b) * β
    fragColor = (xL + xR + xB + xT + uAlpha * bC) * uBeta;
}
