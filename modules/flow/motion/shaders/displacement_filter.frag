#version 460 core

// Displacement Filter Shader
// With z_smooth, x/y through the depth track's alpha–beta filter over z's time, so it lags as z does. The
// positions step by the measured displacement each frame; the state is carried from where the pixel
// was `carry` back, its displacement predicts the step, and the residual corrects it by alpha and
// alpha² / (2 − alpha). The time is z's, the 1€ filter's: `smoothing` at rest, over 1 + speed / `xy_halving`
// in a motion, the speed the largest displacement of the 3×3 past `xy_rest` (0 halving: `smoothing` throughout).
// Out: the filtered displacement (RG, uv per frame) and the position error (BA, uv).

uniform sampler2D tex0;  // The measured displacement (RG, uv per frame)
uniform sampler2D tex1;  // The filter the frame before: displacement RG, position error BA
uniform sampler2D carry; // The displacement the state is carried by, the speed read from (RG, uv per frame)
uniform float smoothing;    // Time followed over at rest (ms); 0 takes the measurement whole
uniform float xy_rest;      // x/y speed below which it is at rest (px per frame)
uniform float xy_halving;   // x/y speed past xy_rest at which the time is half (px per frame); 0 off
uniform float delta_time;   // Seconds since the last frame

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

// The x/y speed here: the largest displacement of the 3×3 texels (px per frame)
float xy_speed() {
    vec2 size = vec2(textureSize(carry, 0));
    ivec2 at = ivec2(floor(texCoord * size));
    ivec2 last_texel = ivec2(size) - 1;
    float largest = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            largest = max(largest, length(texelFetch(carry, clamp(at + ivec2(i, j), ivec2(0), last_texel), 0).xy * size));
        }
    }
    return largest;
}

void main() {
    float alpha = 1.0;
    if (smoothing > 0.0) {
        float follow_time = smoothing;
        if (xy_halving > 0.0) {
            follow_time = smoothing / (1.0 + max(xy_speed() - xy_rest, 0.0) / xy_halving);
        }
        alpha = 1.0 - exp(-delta_time * 1000.0 / follow_time);
    }
    vec4 carried = texture(tex1, texCoord - texture(carry, texCoord).xy);
    float beta = alpha * alpha / (2.0 - alpha);
    vec2 residual = texture(tex0, texCoord).xy - carried.ba - carried.rg;
    fragColor = guard(vec4(carried.rg + beta * residual, -(1.0 - alpha) * residual));
}
