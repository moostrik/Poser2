#version 460 core

// The dual filter (Marius Bjørge, "Bandwidth-Efficient Rendering", SIGGRAPH 2015). The bilinear filter averages four
// texels in each tap placed between them.

uniform sampler2D src;
uniform bool upsample;
uniform float threshold;        // down only: 0 or more keeps what is brighter than it (the bloom's bright pass)
#define halfTexel (0.5 / vec2(textureSize(src, 0)))

in vec2 texCoord;
out vec4 fragColor;

vec4 bright(vec4 c) {
    return threshold < 0.0 ? c : vec4(max(c.rgb - threshold, 0.0), c.a);
}

void main() {
    vec2 h = halfTexel;
    if (!upsample) {
        vec4 sum = bright(texture(src, texCoord)) * 4.0;
        sum += bright(texture(src, texCoord - h));
        sum += bright(texture(src, texCoord + h));
        sum += bright(texture(src, texCoord + vec2(h.x, -h.y)));
        sum += bright(texture(src, texCoord - vec2(h.x, -h.y)));
        fragColor = sum / 8.0;
    } else {
        vec4 sum = texture(src, texCoord + vec2(-h.x * 2.0, 0.0));
        sum += texture(src, texCoord + vec2(-h.x, h.y)) * 2.0;
        sum += texture(src, texCoord + vec2(0.0, h.y * 2.0));
        sum += texture(src, texCoord + vec2(h.x, h.y)) * 2.0;
        sum += texture(src, texCoord + vec2(h.x * 2.0, 0.0));
        sum += texture(src, texCoord + vec2(h.x, -h.y)) * 2.0;
        sum += texture(src, texCoord + vec2(0.0, -h.y * 2.0));
        sum += texture(src, texCoord + vec2(-h.x, -h.y)) * 2.0;
        fragColor = sum / 12.0;
    }
}
