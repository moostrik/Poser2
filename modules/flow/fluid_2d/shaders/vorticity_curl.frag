#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out float fragColor;

// Input textures
uniform sampler2D uVelocity; // Velocity field (RG32F)
uniform sampler2D uObstacle; // Obstacle mask (1.0 = obstacle, 0.0 = fluid)
uniform bool uHasObstacles;  // Skip obstacle checks when false

// Parameters
uniform vec2 uHalfRdxInv; // 0.5 / (the cell × the radius in cells), widths, on x and y
uniform vec2 uRadius;     // the sampling radius in cells on x and y, at least one: the radius in widths over each cell's side

void main() {
    vec2 st = texCoord;

    // Early exit for obstacle pixels
    if (uHasObstacles && texture(uObstacle, st).r > 0.5) {
        fragColor = 0.0;
        return;
    }

    vec2 texelSize = 1.0 / textureSize(uVelocity, 0);

    // Sample at the radius; beyond the field the velocity's border, its wall
    vec2 offsetX = vec2(uRadius.x * texelSize.x, 0.0);
    vec2 offsetY = vec2(0.0, uRadius.y * texelSize.y);

    vec2 vT = texture(uVelocity, st + offsetY).xy;
    vec2 vB = texture(uVelocity, st - offsetY).xy;
    vec2 vR = texture(uVelocity, st + offsetX).xy;
    vec2 vL = texture(uVelocity, st - offsetX).xy;

    fragColor = uHalfRdxInv.x * (vR.y - vL.y) - uHalfRdxInv.y * (vT.x - vB.x);
}
