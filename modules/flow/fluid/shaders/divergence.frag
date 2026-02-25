#version 460 core

// Compute divergence of velocity field
// div(v) = ∂vx/∂x + ∂vy/∂y using central differences

in vec2 texCoord;
out float fragColor;

uniform sampler2D uVelocity;   // Velocity field (RG32F)
uniform sampler2D uObstacle;   // Obstacle mask (R8, CLAMP_TO_BORDER=1)
uniform bool uHasObstacles;    // Skip obstacle checks when false

uniform vec2 uHalfRdxInv;  // (0.5/gridScale_x, 0.5/gridScale_y) for aspect correction

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

    // Central difference divergence (aspect-corrected)
    fragColor = uHalfRdxInv.x * (vR.x - vL.x) + uHalfRdxInv.y * (vT.y - vB.y);
}
