#version 460 core

uniform sampler2D tex0;
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D noise;
uniform float blend1;
uniform float blend2;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 noiseTexel = texture(noise, texCoord);

    float offset = 0.01;

    vec4 texel0 = texture(tex0, texCoord + noiseTexel.rg * offset);
    vec4 texel1 = texture(tex1, texCoord + noiseTexel.gb * offset);
    vec4 texel2 = texture(tex2, texCoord + noiseTexel.rb * offset);
    // noiseTexel = step(0.7, noiseTexel);
    vec4 inv_noise = vec4(1.0 - noiseTexel.rgb, 1.0);

    float w0 = 1.5 - blend1 * blend2;
    float w1 = blend1;
    float w2 = blend2;

    // float w0 = 2.0 * max(noiseTexel.r, inv_noise.r);
    // float w1 = step(0.3, noiseTexel.r) * blend1;
    // float w2 = step(0.3, inv_noise.r) * blend2;

    vec4 color = texel0 * w0 + texel1 * w1 + texel2 * w2;
    float magnitude = length(color.rgb);
    color *= 2.0;
    float t = 2.0;
    if (magnitude > t) {
        color.rgb = (color.rgb / magnitude) * t;
    }


    fragColor = color;
}