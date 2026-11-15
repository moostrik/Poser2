#version 460 core

uniform sampler2D src;
#define texelSize (1.0 / vec2(textureSize(src, 0)))
uniform vec2 direction;     // (1, 0) across, (0, 1) down
uniform float spacing;      // texels between taps, 1 for the kernel as written

in vec2 texCoord;
out vec4 fragColor;

// A Gaussian of 41 texels in 21 taps: each tap but the middle one sits between two texels, so the bilinear filter
// averages the two.
const float weights[10] = float[](0.081149, 0.076883, 0.070266, 0.061948, 0.052683,
                                  0.043219, 0.034202, 0.026109, 0.019227, 0.013658);

void main(void) {
    vec2 step = direction * texelSize * spacing;
    vec3 sum = 0.041312 * texture(src, texCoord).rgb;
    for (int i = 0; i < 10; i++) {
        vec2 offset = step * (float(i) * 2.0 + 1.5);
        sum += weights[i] * texture(src, texCoord + offset).rgb;
        sum += weights[i] * texture(src, texCoord - offset).rgb;
    }
    fragColor = vec4(sum, 1.0);
}
