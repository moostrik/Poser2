#version 460 core

// Weighted Sum 2D Shader
// The 2D bridge's mix: up to seven sources, each times its weight, summed (RG); a source at weight 0 adds nothing.
// With `lengths`, the length of each source's xy times its weight, summed, for an amount (red); a scalar source
// (`scalars`, red alone, the mask) gives its red, read as red: a single-channel texture reads (r, r, r, 1). The
// sources share one unit, so the weights are shares, not gains. With `capped` the sum is held at 1, a presence.

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;
uniform sampler2D tex4;
uniform sampler2D tex5;
uniform sampler2D tex6;
uniform float weights[7];
uniform bool scalars[7];
uniform bool lengths;
uniform bool capped;

in vec2 texCoord;
out vec4 fragColor;

vec2 term(sampler2D tex, float weight, bool scalar) {
    if (weight == 0.0) return vec2(0.0);     // a source at weight 0 is not read, whatever it holds
    if (scalar) return vec2(texture(tex, texCoord).r * weight);
    vec2 value = texture(tex, texCoord).xy;
    return lengths ? vec2(length(value)) * weight : value * weight;
}

void main() {
    vec2 sum = term(tex0, weights[0], scalars[0]) + term(tex1, weights[1], scalars[1]) + term(tex2, weights[2], scalars[2])
             + term(tex3, weights[3], scalars[3]) + term(tex4, weights[4], scalars[4]) + term(tex5, weights[5], scalars[5])
             + term(tex6, weights[6], scalars[6]);
    if (capped) sum = min(sum, vec2(1.0));
    fragColor = vec4(sum, 0.0, 0.0);   // z and w unused
}
