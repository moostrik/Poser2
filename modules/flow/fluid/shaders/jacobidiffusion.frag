#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec4 fragColor;

// Input textures
uniform sampler2D uSource;         // Previous iteration of field to diffuse
uniform sampler2D uObstacle;       // Obstacle mask (1.0 = obstacle, 0.0 = fluid)
uniform sampler2D uObstacleOffset; // Neighbor obstacle flags (RGBA)

// Parameters
uniform vec2 uAlpha;   // (1/dx², 1/dy²) Laplacian weights
uniform float uGamma;  // 1/(ν·Δt) central coefficient
uniform float uBeta;   // 1/(2*alpha_x + 2*alpha_y + gamma)

void main() {
    vec2 st = texCoord;

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

    // Jacobi iteration for anisotropic diffusion:
    // (alpha_x*(xL + xR) + alpha_y*(xB + xT) + gamma*xC) * beta
    fragColor = (uAlpha.x * (xL + xR) + uAlpha.y * (xB + xT) + uGamma * xC) * uBeta;
}
