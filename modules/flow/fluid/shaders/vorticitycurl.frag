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
uniform float uRadius;  // Sampling radius in texels

void main() {
    vec2 st = texCoord;
    vec2 texelSize = 1.0 / textureSize(uVelocity, 0);

    // Sample at radius distance instead of 1 texel
    vec2 offsetX = vec2(uRadius * texelSize.x, 0.0);
    vec2 offsetY = vec2(0.0, uRadius * texelSize.y);

    vec2 vT = texture(uVelocity, st + offsetY).xy;
    vec2 vB = texture(uVelocity, st - offsetY).xy;
    vec2 vR = texture(uVelocity, st + offsetX).xy;
    vec2 vL = texture(uVelocity, st - offsetX).xy;

    // Scale half_rdx by radius (larger radius = larger denominator)
    vec2 scaledHalfRdx = uHalfRdxInv / uRadius;

    fragColor = scaledHalfRdx.x * (vR.y - vL.y) - scaledHalfRdx.y * (vT.x - vB.x);
}
