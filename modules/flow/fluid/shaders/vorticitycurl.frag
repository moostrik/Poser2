#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoordVarying;

// Output
out float fragColor;

// Input textures
uniform sampler2D uVelocity; // Velocity field (RG32F)
uniform sampler2D uObstacle; // Obstacle mask (1.0 = obstacle, 0.0 = fluid)

// Parameters
uniform float uHalfRdx; // 0.5 / gridScale

void main() {
    vec2 st = texCoordVarying;

    // Check if inside obstacle
    if (texture(uObstacle, st).r >= 0.99) {
        fragColor = 0.0;
        return;
    }

    // Sample neighbor velocities
    vec2 vT = textureOffset(uVelocity, st, ivec2(0, 1)).xy;
    vec2 vB = textureOffset(uVelocity, st, ivec2(0, -1)).xy;
    vec2 vR = textureOffset(uVelocity, st, ivec2(1, 0)).xy;
    vec2 vL = textureOffset(uVelocity, st, ivec2(-1, 0)).xy;

    // Compute curl: halfrdx * ((vT.x - vB.x) - (vR.y - vL.y))
    // This is the z-component of the curl (∇ × v)
    fragColor = uHalfRdx * ((vT.x - vB.x) - (vR.y - vL.y));
}
