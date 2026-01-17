#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoordVarying;

// Output
out vec4 fragColor;

// Input textures
uniform sampler2D uSource;         // Previous iteration of field to diffuse
uniform sampler2D uObstacle;       // Obstacle mask (1.0 = obstacle, 0.0 = fluid)
uniform sampler2D uObstacleOffset; // Neighbor obstacle flags (RGBA)

// Parameters
uniform float uAlpha;  // -(gridScale^2) / timestep
uniform float uBeta;   // 1 / (4 + alpha)

void main() {
    vec2 st = texCoordVarying;

    // Check if current pixel is inside obstacle
    float oC = texture(uObstacle, st).r;
    if (oC >= 0.99) {
        fragColor = vec4(0.0);
        return;
    }

    // Sample neighbors with textureOffset
    vec4 xT = textureOffset(uSource, st, ivec2(0, 1));
    vec4 xB = textureOffset(uSource, st, ivec2(0, -1));
    vec4 xR = textureOffset(uSource, st, ivec2(1, 0));
    vec4 xL = textureOffset(uSource, st, ivec2(-1, 0));
    vec4 xC = texture(uSource, st);

    // Boundary conditions: if neighbor is obstacle, use center value
    vec4 oN = texture(uObstacleOffset, st);
    xT = mix(xT, xC, oN.r);  // Top neighbor
    xB = mix(xB, xC, oN.g);  // Bottom neighbor
    xR = mix(xR, xC, oN.b);  // Right neighbor
    xL = mix(xL, xC, oN.a);  // Left neighbor

    // Jacobi iteration for diffusion: (xL + xR + xB + xT + alpha * xC) * beta
    fragColor = (xL + xR + xB + xT + uAlpha * xC) * uBeta;
}
