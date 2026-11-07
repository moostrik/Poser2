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
flat in float vDepth;

uniform bool  uWeighted;        // the weighted look: three targets, the weighted sum, the revealage, the plain sum
uniform float uEmission;
uniform float uAbsorption;
uniform float uTint;

layout(location = 0) out vec4 fragColor;      // the slice's sum; the weighted look's weighted emission and opacity
layout(location = 1) out vec4 fragReveal;     // the weighted look's opacity per channel, multiplied out of 1
layout(location = 2) out vec4 fragAdditive;   // the weighted look's plain emission sum, for the glow

const vec3 WATER_CHAN = vec3(1.8, 1.0, 0.6);

void emit(vec4 c) {
    if (!uWeighted) {
        fragColor = c;
        return;
    }
    // A particle's emission and its opacity per channel; the depth weight puts near ink over far
    // (McGuire and Bavoil 2013), its range kept inside a float's
    vec3 e = uEmission * max(c.rgb, vec3(0.0));
    vec3 alpha = 1.0 - exp(-uAbsorption * mix(vec3(1.0), WATER_CHAN, uTint) * max(c.a, 0.0));
    float a = dot(alpha, vec3(1.0 / 3.0));
    float w = clamp(0.03 / (1e-5 + pow(vDepth + 0.05, 4.0)), 1e-2, 3e3);
    fragColor = vec4(e * w, a * w);
    fragReveal = vec4(alpha, 0.0);
    fragAdditive = vec4(e, 0.0);
}

void main() {
    if (uMode != 2) {
        emit(vColor);
        return;
    }
    vec2 d = (gl_PointCoord - 0.5) * vPointSize;
    float s2 = vSigma * vSigma;
    float r2 = dot(d, d);
    if (r2 > CUT * CUT * s2) discard;
    float w = exp(-r2 / (2.0 * s2)) / (6.2831853 * s2 * INSIDE);
    emit(vColor * w);
}
