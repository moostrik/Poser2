#version 460 core

/*
 * MacCormack2D - the MacCormack correction of a semi-Lagrangian advection (Selle et al. 2008), as MacCormack3D on
 * the plane.
 *
 * The predictor (uForward) is the semi-Lagrangian step of the source, traced back along the velocity. Traced
 * forward again from each cell, the predictor gives back the source up to the step's error; half of that error is
 * added back:
 *
 *     phi = forward(x) + 0.5 * (source(x) - forward(x + dt * v(x)))
 *
 * and bounded by the source's four texels around the backward trace: where it leaves them, the predictor stands
 * (Selle's revert), or, with uClamp (the field's `limiter`), the corrected value is cut to them. A trace that reads
 * past a wall keeps the predictor: the wall's value is in no texel. Then the dissipation and the evaporation, as the
 * plain step applies them, and the guard. The velocity and the obstacle are sampled by uv, so the field may sit on a
 * finer grid than they do.
 */

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D uVelocity;   // Velocity field (RG16F), widths/s
uniform sampler2D uSource;     // The field before the step
uniform sampler2D uForward;    // The semi-Lagrangian predictor
uniform sampler2D uObstacle;   // Obstacle mask (R8)
uniform bool uHasObstacles;

uniform float uTimestep;       // dt * advection_scale
uniform vec2  uRdx;            // (1, aspect): uv per width on x and y
uniform float uDissipation;    // the fade's multiplier this step; 1 = none
uniform float uEvaporation;    // dt / evaporation_time: a full value's share lost this step well above the knee; 0 off
uniform float uEvaporationKnee; // the size below which the loss turns from a fixed amount to a share
uniform int   uMeasure;        // how the evaporation reads a field's size: 0 its alpha, 1 its vector's length, 2 |r|
uniform int   uGuardSlot;      // the field's slot in the stage's FiniteGuard; -1 no guard
uniform bool  uClamp;          // where the correction overshoots: cut to the bounds, else revert to the predictor

#include "walls_2d.glsl"

// The stage's non-finite counts (FiniteGuard, GUARD_BINDING)
layout(std430, binding = 7) buffer FiniteGuard { uint uNonFinite[]; };

vec4 evaporate(vec4 value) {
    float size = uMeasure == 0 ? value.a : (uMeasure == 1 ? length(value.xy) : abs(value.r));
    if (size <= 0.0) return value;
    return value * exp(-uEvaporation / (size + uEvaporationKnee));
}

bool non_finite(vec4 value) {
    return any(equal(floatBitsToUint(value) & 0x7F800000u, uvec4(0x7F800000u)));
}

void main() {
    ivec2 gid = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(uSource, 0);
    vec2 st = texCoord;
    vec4 source = texelFetch(uSource, gid, 0);

    if (uGuardSlot >= 0 && non_finite(source)) {           // a non-finite value held: 0, counted, not traced
        atomicAdd(uNonFinite[uGuardSlot], 1u);
        fragColor = vec4(0.0);
        return;
    }
    vec4 forward = texelFetch(uForward, gid, 0);

    if (uHasObstacles && texture(uObstacle, st).r > 0.5) {
        fragColor = source;
        return;
    }

    vec2 step = uTimestep * uRdx * texture(uVelocity, st).xy;
    vec2 st_back = st - step;
    vec4 back = texture(uForward, st + step);
    vec4 result = forward + 0.5 * (source - back);

    // A trace that reads past a wall read the wall's value, which no texel holds: the predictor stands
    ivec2 base = ivec2(floor(st_back * vec2(size) - 0.5));
    if (any(lessThan(base, ivec2(0))) || any(greaterThan(base + 1, size - 1))) {
        result = forward;
    } else {
        // The limiter: the source's four texels the backward trace read between
        vec4 lo = vec4(1e30);
        vec4 hi = vec4(-1e30);
        for (int k = 0; k < 4; k++) {
            vec4 value = texelFetch(uSource, base + ivec2(k & 1, (k >> 1) & 1), 0);
            lo = min(lo, value);
            hi = max(hi, value);
        }
        if (any(greaterThan(lo, hi))) {
            result = forward;                                   // no finite texel: no bounds
        } else if (uClamp) {
            result = clamp(result, lo, hi);
        } else {
            bool out_of_bounds = any(lessThan(result, lo)) || any(greaterThan(result, hi));
            result = out_of_bounds ? forward : result;
        }
    }

    result *= uDissipation;
    if (uEvaporation > 0.0) {
        result = evaporate(result);
    }
    // The velocity against a closed wall braked along it (walls_2d.glsl; uWallKeep 1 for every other field)
    result.xy = slip(result.xy, ivec2(gl_FragCoord.xy), textureSize(uSource, 0));
    if (uGuardSlot >= 0 && non_finite(result)) {
        atomicAdd(uNonFinite[uGuardSlot], 1u);
        result = vec4(0.0);
    }
    fragColor = result;
}
