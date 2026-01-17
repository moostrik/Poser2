#version 460 core

// Compute divergence of velocity field
// div(v) = ∂vx/∂x + ∂vy/∂y using central differences

in vec2 texCoord;
out float fragColor;

uniform sampler2D uVelocity;        // Velocity field (RG32F)
uniform sampler2D uObstacle;        // Obstacle mask (R8/R32F)
uniform sampler2D uObstacleOffset;  // Neighbor obstacle info (RGBA8)

uniform float uHalfRdx;  // 0.5 / gridScale

void main() {
    vec2 st = texCoord;

    float obstacle = texture(uObstacle, st).x;
    if (obstacle > 0.5) {
        fragColor = 0.0;
        return;
    }

    // Sample velocity at neighbors
    vec2 vT = textureOffset(uVelocity, st, ivec2(0, 1)).xy;
    vec2 vB = textureOffset(uVelocity, st, ivec2(0, -1)).xy;
    vec2 vR = textureOffset(uVelocity, st, ivec2(1, 0)).xy;
    vec2 vL = textureOffset(uVelocity, st, ivec2(-1, 0)).xy;
    vec2 vC = texture(uVelocity, st).xy;

    // No-slip boundary conditions (zero velocity at obstacles)
    // Use -vC if neighbor is obstacle
    vec4 oN = texture(uObstacleOffset, st);
    vT = mix(vT, -vC, oN.x);
    vB = mix(vB, -vC, oN.y);
    vR = mix(vR, -vC, oN.z);
    vL = mix(vL, -vC, oN.w);

    // Central difference divergence
    fragColor = uHalfRdx * ((vR.x - vL.x) + (vT.y - vB.y));
}
