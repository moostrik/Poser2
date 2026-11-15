#version 460 core

uniform sampler2D src;
uniform sampler2D dst;
uniform sampler2D gradient;
uniform float progress;     // 0: all src, 1: all dst
uniform float softness;     // the edge's width, in gradient values

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float g = texture(gradient, texCoord).r;
    // The edge runs from g to g + softness; progress is stretched so that 1 passes every edge.
    float t = progress * (1.0 + softness);
    float amount = softness > 0.0 ? clamp((t - g) / softness, 0.0, 1.0) : step(g, progress);
    fragColor = mix(texture(src, texCoord), texture(dst, texCoord), amount);
}
