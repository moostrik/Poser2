#version 460 core

uniform sampler2D src;
uniform sampler2D blueNoise;    // 64 x 64 thresholds 0..1
uniform float levels;           // the output's steps per channel: 255 for 8 bits
uniform int pattern;            // DitherPattern: 0 ordered, 1 blue noise

in vec2 texCoord;
out vec4 fragColor;

// The 8 x 8 Bayer matrix's threshold at a pixel, 0..1, built from its bits
float bayer8(ivec2 p) {
    int x = p.x & 7;
    int y = p.y & 7;
    int xy = x ^ y;
    int value = ((xy & 1) << 5) | ((x & 1) << 4) | ((xy & 2) << 2) | ((x & 2) << 1) | ((xy & 4) >> 1) | ((x & 4) >> 2);
    return (float(value) + 0.5) / 64.0;
}

void main() {
    vec4 color = texture(src, texCoord);
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float threshold = pattern == 1 ? texelFetch(blueNoise, pixel & 63, 0).r : bayer8(pixel);
    fragColor = vec4(floor(color.rgb * levels + threshold) / levels, color.a);
}
