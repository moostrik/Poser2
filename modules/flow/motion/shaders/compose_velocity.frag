#version 460 core

// Compose Velocity Shader
// A flow's 3D velocity: its x/y and its z, each times its strength, 0 where it is not finite, counted by
// the guard: the bridges feed back, so one such pixel would stay and spread through their blur.
// With has_mask, times the mask; with has_depth_mask, z also times the depth mask, where the depth is the player's

uniform sampler2D tex0;  // x/y RG
uniform sampler2D tex1;  // z R, read only with has_depth
uniform sampler2D tex2;  // mask R (0 to 1), read only with has_mask
uniform sampler2D tex3;  // depth mask R (0 to 1), read only with has_depth_mask
uniform bool has_depth;
uniform bool has_mask;
uniform bool has_depth_mask;
uniform vec3 strength;   // Multiplier on x, y and z

in vec2 texCoord;
out vec4 fragColor;

// The guard (FiniteGuard): a value whose exponent bits are all ones is an infinity or a NaN; it becomes 0 and is counted
layout(std430, binding = 7) buffer FiniteGuard { uint uNonFinite[]; };
uniform int uGuardSlot;  // The output's slot in the motion's FiniteGuard; -1 no guard

vec4 guard(vec4 value) {
    if (uGuardSlot < 0 || !any(equal(floatBitsToUint(value) & 0x7F800000u, uvec4(0x7F800000u)))) return value;
    atomicAdd(uNonFinite[uGuardSlot], 1u);
    return vec4(0.0);
}

void main() {
    vec3 velocity = vec3(texture(tex0, texCoord).xy, has_depth ? texture(tex1, texCoord).r : 0.0) * strength;
    if (has_mask) {
        velocity *= texture(tex2, texCoord).r;
    }
    if (has_depth_mask) {
        velocity.z *= texture(tex3, texCoord).r;
    }
    fragColor = guard(vec4(velocity, 0.0));     // w unused
}
