#version 460 core

// The vorticity confinements' force × dt at the cells, summed, as VorticityForce3D on the plane (Fedkiw 2001):
//   f = ε h (N × ω) Δt, N = ∇|ω| / |∇|ω||, ω along z: N × ω = (N.y ω, −N.x ω)
// each confinement from its own curl (its channel, VorticityCurl) and strength. Beyond the walls the curl's edge, this
// cell's (the texture's CLAMP_TO_EDGE), so no slope toward the wall for the force to push along, as the 3D's.

in vec2 texCoord;
out vec2 fragColor;

const int MAX_CURLS = 3;     // VorticityCurl.MAX_CURLS

layout(binding = 0) uniform sampler2D uCurl;     // each confinement's curl, one channel each (RGBA16F)
layout(binding = 1) uniform sampler2D uObstacle; // Obstacle mask (R8, CLAMP_TO_BORDER=1)
uniform bool uHasObstacles;  // Skip obstacle checks when false

uniform vec2  uHalfRdxInv;               // 0.5 / the cell, widths, on x and y
uniform int   uCount;                    // the confinements, 1 to MAX_CURLS
uniform float uTimestep[MAX_CURLS];      // each one's strength epsilon × the cell's width h × dt

void main() {
    vec2 st = texCoord;

    if (uHasObstacles && texture(uObstacle, st).r > 0.5) {
        fragColor = vec2(0.0);
        return;
    }

    vec4 cT = abs(textureOffset(uCurl, st, ivec2(0, 1)));
    vec4 cB = abs(textureOffset(uCurl, st, ivec2(0, -1)));
    vec4 cR = abs(textureOffset(uCurl, st, ivec2(1, 0)));
    vec4 cL = abs(textureOffset(uCurl, st, ivec2(-1, 0)));
    vec4 cC = texture(uCurl, st);

    vec2 force = vec2(0.0);
    for (int k = 0; k < uCount; k++) {
        vec2 gradient = vec2(uHalfRdxInv.x * (cR[k] - cL[k]), uHalfRdxInv.y * (cT[k] - cB[k]));
        float len = length(gradient);
        vec2 N = (len > 1e-6) ? gradient / len : vec2(0.0);
        force += uTimestep[k] * cC[k] * vec2(N.y, -N.x);
    }
    fragColor = force;
}
