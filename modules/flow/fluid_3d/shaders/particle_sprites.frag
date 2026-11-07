#version 460 core

// ParticleSprites: a point or a streak's pixel takes the particle's value as it is; a blob's pixel the
// value times a normalized gaussian of its sigma, cut at CUT sigma and scaled by the share inside, so a blob
// sums to its value whatever its size.

uniform int uMode;              // 0 points, 1 streaks, 2 blobs

const float CUT = 2.5;          // sigma; as the vertex shader's
const float INSIDE = 0.95606;   // 1 - exp(-CUT² / 2), the 2D gaussian's share inside CUT sigma

flat in vec4 vColor;
flat in float vSigma;
flat in float vPointSize;

layout(location = 0) out vec4 fragColor;

void main() {
    if (uMode != 2) {
        fragColor = vColor;
        return;
    }
    vec2 d = (gl_PointCoord - 0.5) * vPointSize;
    float s2 = vSigma * vSigma;
    float r2 = dot(d, d);
    if (r2 > CUT * CUT * s2) discard;
    float w = exp(-r2 / (2.0 * s2)) / (6.2831853 * s2 * INSIDE);
    fragColor = vColor * w;
}
