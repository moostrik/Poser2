#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec2 fragColor;

// Input textures
uniform sampler2D uCurl; // Curl magnitude field (R32F)

// Parameters
uniform vec2 uHalfRdxInv;  // (0.5/gridScale_x, 0.5/gridScale_y) for aspect correction
uniform float uTimestep; // Vorticity confinement timestep

void main() {
    vec2 st = texCoord;

    // Sample curl neighbors (take absolute value)
    float cT = abs(textureOffset(uCurl, st, ivec2(0, 1)).r);
    float cB = abs(textureOffset(uCurl, st, ivec2(0, -1)).r);
    float cR = abs(textureOffset(uCurl, st, ivec2(1, 0)).r);
    float cL = abs(textureOffset(uCurl, st, ivec2(-1, 0)).r);
    float cC = texture(uCurl, st).r;

    // Compute gradient of curl magnitude (aspect-corrected)
    // X: vertical difference (cT - cB), Y: horizontal difference (cR - cL)
    vec2 dw = vec2(uHalfRdxInv.x * (cR - cL), uHalfRdxInv.y * (cT - cB));

    // Normalize gradient (add epsilon to avoid division by zero)
    dw = normalize(dw + 0.000001);

    // Flip X and negate Y for vorticity confinement direction
    vec2 force = vec2(dw.y, -dw.x);

    // Vorticity confinement force: direction * curl * timestep
    fragColor = force * cC * uTimestep;
}
