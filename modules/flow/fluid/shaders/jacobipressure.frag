#version 460 core

// Jacobi iterative solver for Poisson pressure equation
// Solves: ∇²p = -∇·v

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D uSource;      // Previous pressure estimate (R32F)
uniform sampler2D uDivergence;  // Velocity divergence (R32F)
uniform sampler2D uObstacle;    // Obstacle mask (R8, CLAMP_TO_BORDER=1)

uniform vec2 uAlpha;  // -(gridScale²)
uniform float uBeta;   // 0.25 (= 1/4 for 4 neighbors)

void main() {
    vec2 st = texCoord;

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

    // Inline neighbor obstacle sampling
    float oT = textureOffset(uObstacle, st, ivec2(0, 1)).r;
    float oB = textureOffset(uObstacle, st, ivec2(0, -1)).r;
    float oR = textureOffset(uObstacle, st, ivec2(1, 0)).r;
    float oL = textureOffset(uObstacle, st, ivec2(-1, 0)).r;

    // Neumann boundary conditions: zero gradient at obstacles
    xT = mix(xT, xC, oT);
    xB = mix(xB, xC, oB);
    xR = mix(xR, xC, oR);
    xL = mix(xL, xC, oL);

    // Jacobi iteration: x^(k+1) = (xL + xR + xB + xT + α*b) * β
    fragColor = (uAlpha.x * (xL + xR) + uAlpha.y * (xB + xT) - bC) * uBeta;
}
