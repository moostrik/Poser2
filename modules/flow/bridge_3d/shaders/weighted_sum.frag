#version 460 core

// Weighted Sum Shader
// The bridge's mix: up to six sources, each times its weight, summed (RGB); a source at weight 0 adds
// nothing. With `lengths`, the length of each source's xyz times its weight, summed, for an amount.
// The sources share one unit, so the weights are shares, not gains

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;
uniform sampler2D tex4;
uniform sampler2D tex5;
uniform float weights[6];
uniform bool lengths;

in vec2 texCoord;
out vec4 fragColor;

vec3 term(sampler2D tex, float weight) {
    if (weight == 0.0) return vec3(0.0);     // a source at weight 0 is not read, whatever it holds
    vec3 value = texture(tex, texCoord).rgb;
    return lengths ? vec3(length(value)) * weight : value * weight;
}

void main() {
    vec3 sum = term(tex0, weights[0]) + term(tex1, weights[1]) + term(tex2, weights[2])
             + term(tex3, weights[3]) + term(tex4, weights[4]) + term(tex5, weights[5]);
    fragColor = vec4(sum, 0.0);   // w unused
}
