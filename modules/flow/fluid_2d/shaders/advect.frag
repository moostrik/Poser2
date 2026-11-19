#version 460 core

// Semi-Lagrangian advection with the fade, the evaporation and the guard, as the 3D's
// Backward trace along velocity field and sample source
//
// Advection formula:  st_back = st - timestep * rdx * velocity, the velocity in widths/s
//   timestep = dt * advection_scale
//   rdx = vec2(1.0, aspect): uv per width, the field's width over its height on y

in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D uVelocity;   // Velocity field (RG16F), widths/s
uniform sampler2D uSource;     // Field to advect (any format)
uniform sampler2D uObstacle;   // Obstacle mask (R8 or R32F)
uniform bool uHasObstacles;    // Skip obstacle checks when false

uniform float uTimestep;       // dt * advection_scale
uniform vec2  uRdx;            // (1, aspect): uv per width on x and y
uniform float uDissipation;    // the fade's multiplier this step; 1 = none
uniform float uEvaporation;    // dt / evaporation_time: a full value's share lost this step well above the knee; 0 off
uniform float uEvaporationKnee; // the size below which the loss turns from a fixed amount to a share
uniform int   uMeasure;        // how the evaporation reads a field's size: 0 its alpha, 1 its vector's length, 2 |r|
uniform int   uGuardSlot;      // the field's slot in the stage's FiniteGuard; -1 no guard

#include "walls_2d.glsl"

// The stage's non-finite counts (FiniteGuard, GUARD_BINDING)
layout(std430, binding = 7) buffer FiniteGuard { uint uNonFinite[]; };

// The evaporation, a saturating loss: the field's size ρ times exp(−uEvaporation / (ρ + knee)), every channel
// scaled with it, so a color keeps its hue, a velocity its direction and a signed value its sign; its size the
// alpha of a premultiplied density (0), the length of a vector (1), or the first channel's magnitude (2)
vec4 evaporate(vec4 value) {
    float size = uMeasure == 0 ? value.a : (uMeasure == 1 ? length(value.xy) : abs(value.r));
    if (size <= 0.0) return value;
    return value * exp(-uEvaporation / (size + uEvaporationKnee));
}

// The guard: a value whose exponent bits are all ones is an infinity or a NaN; it becomes 0 and is counted.
// A cell is counted once: where it held one (before the step reads it) or where the step made one
bool non_finite(vec4 value) {
    return any(equal(floatBitsToUint(value) & 0x7F800000u, uvec4(0x7F800000u)));
}

void main() {
    vec2 st = texCoord;

    if (uGuardSlot >= 0 && non_finite(texture(uSource, st))) {
        atomicAdd(uNonFinite[uGuardSlot], 1u);
        fragColor = vec4(0.0);
        return;
    }

    // Obstacle check — write current source value (not black) so bilinear
    // sampling across the boundary extrapolates smoothly instead of bleeding zeros.
    if (uHasObstacles) {
        float obstacle = texture(uObstacle, st).x;
        if (obstacle > 0.5) {
            fragColor = texture(uSource, st);
            return;
        }
    }

    // Backward trace along velocity (aspect-corrected)
    vec2 velocity = texture(uVelocity, st).xy;
    vec2 st_back = st - uTimestep * uRdx * velocity;

    // If backtrace lands inside an obstacle, fall back to current position
    // (no advection) to prevent sampling from zeroed obstacle interior.
    if (uHasObstacles && texture(uObstacle, st_back).x > 0.5) {
        st_back = st;
    }

    // Sample, fade and evaporate
    vec4 result = uDissipation * texture(uSource, st_back);
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
