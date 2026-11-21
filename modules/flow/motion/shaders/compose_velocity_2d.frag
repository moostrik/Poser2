#version 460 core

// Compose Velocity 2D Shader
// A flow's 2D velocity: its x/y times its strength, 0 where it is not finite, counted by the guard: the bridges feed
// back, so one such pixel would stay and spread through their blur. With has_mask, times the mask

layout(binding = 0) uniform sampler2D tex0;  // x/y RG
layout(binding = 1) uniform sampler2D tex1; // mask R (0 to 1), read only with has_mask
uniform bool has_mask;
uniform vec2 strength;   // Multiplier on x and y

in vec2 texCoord;
out vec4 fragColor;

// The guard (FiniteGuard): a value whose exponent bits are all ones is an infinity or a NaN; it becomes 0 and is counted
layout(std430, binding = 7) buffer FiniteGuard { uint uNonFinite[]; };
uniform int uGuardSlot;  // The output's slot in the motion's FiniteGuard; -1 no guard

vec2 guard(vec2 value) {
    if (uGuardSlot < 0 || !any(equal(floatBitsToUint(value) & 0x7F800000u, uvec2(0x7F800000u)))) return value;
    atomicAdd(uNonFinite[uGuardSlot], 1u);
    return vec2(0.0);
}

void main() {
    vec2 velocity = texture(tex0, texCoord).xy * strength;
    if (has_mask) {
        velocity *= texture(tex1, texCoord).r;
    }
    fragColor = vec4(guard(velocity), 0.0, 0.0);     // z and w unused
}
