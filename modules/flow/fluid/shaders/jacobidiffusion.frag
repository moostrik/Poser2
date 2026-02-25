#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec4 fragColor;

// Input textures
uniform sampler2D uSource;    // Previous iteration of field to diffuse
uniform sampler2D uObstacle;  // Obstacle mask (R8, CLAMP_TO_BORDER=1)

// Parameters
uniform vec2 uAlpha;   // (1/dx², 1/dy²) Laplacian weights
uniform float uGamma;  // 1/(ν·Δt) central coefficient
uniform float uBeta;   // 1/(2*alpha_x + 2*alpha_y + gamma)

void main() {
    vec2 st = texCoord;

    // Check if current pixel is inside obstacle
    float oC = texture(uObstacle, st).r;
    if (oC > 0.5) {
        fragColor = vec4(0.0);
        return;
    }

    // Sample neighbors with textureOffset
    vec4 xT = textureOffset(uSource, st, ivec2(0, 1));
    vec4 xB = textureOffset(uSource, st, ivec2(0, -1));
    vec4 xR = textureOffset(uSource, st, ivec2(1, 0));
    vec4 xL = textureOffset(uSource, st, ivec2(-1, 0));
    vec4 xC = texture(uSource, st);

    // Inline neighbor obstacle sampling
    float oT = textureOffset(uObstacle, st, ivec2(0, 1)).r;
    float oB = textureOffset(uObstacle, st, ivec2(0, -1)).r;
    float oR = textureOffset(uObstacle, st, ivec2(1, 0)).r;
    float oL = textureOffset(uObstacle, st, ivec2(-1, 0)).r;

    // Neumann boundary conditions: zero gradient at obstacles
    xT = mix(xT, xC, oT);
    xB = mix(xB, xC, oB);
    xR = mix(xR, xC, oR);
    xL = mix(xL, xC, oL);

    // Jacobi iteration for anisotropic diffusion:
    // (alpha_x*(xL + xR) + alpha_y*(xB + xT) + gamma*xC) * beta
    fragColor = (uAlpha.x * (xL + xR) + uAlpha.y * (xB + xT) + uGamma * xC) * uBeta;
}
