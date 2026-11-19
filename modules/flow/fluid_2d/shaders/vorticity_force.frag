#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec2 fragColor;

// Input textures
uniform sampler2D uCurl; // Curl magnitude field (R32F)
uniform sampler2D uObstacle; // Obstacle mask (R8, CLAMP_TO_BORDER=1)
uniform bool uHasObstacles;  // Skip obstacle checks when false

// Parameters
uniform vec2 uHalfRdxInv;  // 0.5 / the cell, widths, on x and y
uniform float uTimestep;   // the strength epsilon × the cell's width h × dt (Fedkiw 2001)

void main() {
    vec2 st = texCoord;

    // Early exit for obstacle pixels
    if (uHasObstacles && texture(uObstacle, st).r > 0.5) {
        fragColor = vec2(0.0);
        return;
    }

    // Sample curl neighbors (take absolute value); beyond the walls the curl's edge, this cell's, so no slope
    // toward the wall for the force to push along, as the 3D's
    float cT = abs(textureOffset(uCurl, st, ivec2(0, 1)).r);
    float cB = abs(textureOffset(uCurl, st, ivec2(0, -1)).r);
    float cR = abs(textureOffset(uCurl, st, ivec2(1, 0)).r);
    float cL = abs(textureOffset(uCurl, st, ivec2(-1, 0)).r);
    float cC = texture(uCurl, st).r;

    // Compute gradient of curl magnitude (aspect-corrected)
    // X: vertical difference (cT - cB), Y: horizontal difference (cR - cL)
    vec2 dw = vec2(uHalfRdxInv.x * (cR - cL), uHalfRdxInv.y * (cT - cB));

    // Normalize
    float len = length(dw);
    dw = (len > 1e-6) ? dw / len : vec2(0.0);

    // Flip X and negate Y for vorticity confinement direction
    vec2 force = vec2(dw.y, -dw.x);

    // The confinement f = epsilon h (N × curl) dt, the curl along z
    fragColor = force * cC * uTimestep;
}
