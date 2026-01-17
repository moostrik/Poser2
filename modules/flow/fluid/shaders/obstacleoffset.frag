#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoordVarying;

// Output
out vec4 fragColor;

// Input texture
uniform sampler2D uObstacle; // Obstacle mask (R8: 1.0 = obstacle, 0.0 = fluid)

void main() {
    vec2 st = texCoordVarying;

    // Sample neighbor obstacle values
    // Output RGBA encodes: R=top, G=bottom, B=right, A=left
    // 1.0 = neighbor is obstacle, 0.0 = neighbor is fluid
    fragColor = vec4(
        textureOffset(uObstacle, st, ivec2(0, 1)).r,   // Top
        textureOffset(uObstacle, st, ivec2(0, -1)).r,  // Bottom
        textureOffset(uObstacle, st, ivec2(1, 0)).r,   // Right
        textureOffset(uObstacle, st, ivec2(-1, 0)).r   // Left
    );
}
