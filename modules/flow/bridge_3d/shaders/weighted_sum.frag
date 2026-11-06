#version 460 core

// Weighted Sum Shader
// The bridge's mix: up to five sources, each times its weight, summed (RGB); a source at weight 0 adds
// nothing. The sources share one unit, image widths per second, so the weights are shares, not gains

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;
uniform sampler2D tex4;
uniform float weights[5];

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 sum = texture(tex0, texCoord).rgb * weights[0]
             + texture(tex1, texCoord).rgb * weights[1]
             + texture(tex2, texCoord).rgb * weights[2]
             + texture(tex3, texCoord).rgb * weights[3]
             + texture(tex4, texCoord).rgb * weights[4];
    fragColor = vec4(sum, 0.0);   // w unused
}
