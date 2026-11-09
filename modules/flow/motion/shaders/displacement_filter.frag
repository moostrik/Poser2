#version 460 core

// Displacement Filter Shader
// With xy_smooth, x/y through the depth track's alpha–beta filter, so it lags as z does. The
// positions step by the measured displacement each frame; the state is carried from where the pixel
// was `carry` back, its displacement predicts the step, and the residual corrects it by alpha and
// alpha² / (2 − alpha). Out: the filtered displacement (RG, uv per frame) and the position error (BA, uv).
// The numpy reference is lucas_kanade_flow.filter_displacement

uniform sampler2D tex0;  // The measured displacement (RG, uv per frame)
uniform sampler2D tex1;  // The filter the frame before: displacement RG, position error BA
uniform sampler2D carry; // The displacement the state is carried by (RG, uv per frame)
uniform float alpha;     // Weight of the measurement on the position

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
    vec4 carried = texture(tex1, texCoord - texture(carry, texCoord).xy);
    float beta = alpha * alpha / (2.0 - alpha);
    vec2 residual = texture(tex0, texCoord).xy - carried.ba - carried.rg;
    fragColor = guard(vec4(carried.rg + beta * residual, -(1.0 - alpha) * residual));
}
