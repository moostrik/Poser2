#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out float fragColor;

// Input textures
uniform sampler2D uVelocity; // Velocity field (RG32F)
uniform sampler2D uObstacle; // Obstacle mask (1.0 = obstacle, 0.0 = fluid)

// Parameters
uniform vec2 uHalfRdxInv; // (0.5/gridScale_x, 0.5/gridScale_y) for aspect correction

void main() {
    vec2 st = texCoord;

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

    // Compute curl: ∂vy/∂x - ∂vx/∂y (aspect-corrected)
    // This is the z-component of the curl (∇ × v)
    fragColor = uHalfRdxInv.x * (vT.x - vB.x) - uHalfRdxInv.y * (vR.y - vL.y);
}
