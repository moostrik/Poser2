#version 460 core

#include "color.glsl"

layout(binding = 0) uniform sampler2D src;
uniform float threshold;
uniform float softness;
uniform bool invert;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    float alpha = smoothstep(threshold, threshold + softness, luminance(color.rgb));
    if (invert) alpha = 1.0 - alpha;
    fragColor = vec4(color.rgb, color.a * alpha);
}
