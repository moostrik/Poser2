#version 460 core

// Jacobi iterative solver for Poisson pressure equation
// Solves: ∇²p = -∇·v

in vec2 texCoord;
out vec4 fragColor;

layout(binding = 0) uniform sampler2D uSource;      // Previous pressure estimate (R32F)
layout(binding = 1) uniform sampler2D uDivergence;  // Velocity divergence (R32F)
layout(binding = 2) uniform sampler2D uObstacle;   // Obstacle mask (R8, CLAMP_TO_BORDER=1)
uniform bool uHasObstacles;     // Skip obstacle checks when false

#include "walls_2d.glsl"

uniform vec2 uAlpha;  // -(gridScale²)
uniform float uBeta;   // 1 / (2 αx + 2 αy + s), s the screened equation's 1 / reach² (0 none)

void main() {
    vec2 st = texCoord;

    if (uHasObstacles) {
        float obstacle = texture(uObstacle, st).x;
        if (obstacle > 0.5) {
            fragColor = vec4(0.0);
            return;
        }
    }

    // Sample divergence (RHS of equation)
    vec4 bC = texture(uDivergence, st);

    // Sample pressure at neighbors
    vec4 xT = textureOffset(uSource, st, ivec2(0, 1));
    vec4 xB = textureOffset(uSource, st, ivec2(0, -1));
    vec4 xR = textureOffset(uSource, st, ivec2(1, 0));
    vec4 xL = textureOffset(uSource, st, ivec2(-1, 0));
    vec4 xW = vec4(wall_pressure(texture(uSource, st).x));
    // Beyond the field the wall's ghost (walls_2d.glsl)
    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(uSource, 0);
    if (cell.y + 1 >= size.y) xT = xW;
    if (cell.y - 1 < 0)       xB = xW;
    if (cell.x + 1 >= size.x) xR = xW;
    if (cell.x - 1 < 0)       xL = xW;

    if (uHasObstacles) {
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
    }

    // Jacobi iteration: x^(k+1) = (xL + xR + xB + xT + α*b) * β
    fragColor = (uAlpha.x * (xL + xR) + uAlpha.y * (xB + xT) - bC) * uBeta;
}
