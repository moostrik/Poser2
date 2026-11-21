#version 460 core

// Compute divergence of velocity field
// div(v) = ∂vx/∂x + ∂vy/∂y using central differences, the velocity in widths/s over the cell in widths.
// With the expansion, its rate × the expansion is taken off (Nguyen, Fedkiw and Jensen 2002): the projection makes the
// divergence that, not 0; with closed walls its mean over the water's cells first (uMean, ExpansionMean2D's), as the
// solve with closed walls needs the divergence to sum to 0.

in vec2 texCoord;
out float fragColor;

layout(binding = 0) uniform sampler2D uVelocity;   // Velocity field (RG32F)
layout(binding = 1) uniform sampler2D uObstacle;   // Obstacle mask (R8, CLAMP_TO_BORDER=1)
uniform bool uHasObstacles;    // Skip obstacle checks when false
layout(binding = 2) uniform sampler2D uExpansion; // The expansion, a presence 0 to 1 (R16F); read with uHasExpansion
uniform bool uHasExpansion;
uniform float uRate;           // The expansion's rate, 1/s
uniform bool uSubtractMean;    // The expansion's mean (uMean) taken off first
layout(std430, binding = 1) readonly buffer Mean { float uMean; };

uniform vec2 uHalfRdxInv;  // 1 / (2 × the cell), widths, on x and y

#include "walls_2d.glsl"

void main() {
    vec2 st = texCoord;

    if (uHasObstacles) {
        float obstacle = texture(uObstacle, st).x;
        if (obstacle > 0.5) {
            fragColor = 0.0;
            return;
        }
    }

    // Sample velocity at neighbors
    vec2 vT = textureOffset(uVelocity, st, ivec2(0, 1)).xy;
    vec2 vB = textureOffset(uVelocity, st, ivec2(0, -1)).xy;
    vec2 vR = textureOffset(uVelocity, st, ivec2(1, 0)).xy;
    vec2 vL = textureOffset(uVelocity, st, ivec2(-1, 0)).xy;

    // Beyond the field the wall's ghost (walls_2d.glsl): closed, nothing through the wall's face
    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(uVelocity, 0);
    vec2 vCell = texture(uVelocity, st).xy;
    if (cell.y + 1 >= size.y) vT = wall_velocity(vCell, 1);
    if (cell.y - 1 < 0)       vB = wall_velocity(vCell, 1);
    if (cell.x + 1 >= size.x) vR = wall_velocity(vCell, 0);
    if (cell.x - 1 < 0)       vL = wall_velocity(vCell, 0);

    if (uHasObstacles) {
        vec2 vC = texture(uVelocity, st).xy;

        // Inline neighbor obstacle sampling
        float oT = textureOffset(uObstacle, st, ivec2(0, 1)).r;
        float oB = textureOffset(uObstacle, st, ivec2(0, -1)).r;
        float oR = textureOffset(uObstacle, st, ivec2(1, 0)).r;
        float oL = textureOffset(uObstacle, st, ivec2(-1, 0)).r;

        // No-slip boundary conditions: reflect velocity at obstacles
        vT = mix(vT, -vC, oT);
        vB = mix(vB, -vC, oB);
        vR = mix(vR, -vC, oR);
        vL = mix(vL, -vC, oL);
    }

    // Central difference divergence
    float divergence = uHalfRdxInv.x * (vR.x - vL.x) + uHalfRdxInv.y * (vT.y - vB.y);
    if (uHasExpansion) {
        float expansion = textureLod(uExpansion, st, 0.0).r - (uSubtractMean ? uMean : 0.0);
        divergence -= uRate * expansion;
    }
    fragColor = divergence;
}
