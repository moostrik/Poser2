#version 460 core

#include "color.glsl"

layout(binding = 0) uniform sampler2D src;
uniform vec3 color;
uniform float strength;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 texel = texture(src, texCoord);
    vec3 srcHsv = rgb2hsv(texel.rgb);
    vec3 targetHsv = rgb2hsv(color);

    // Hue toward the color's, saturation raised toward it; the value kept
    float hue = mix(srcHsv.x, targetHsv.x, strength);
    float saturation = mix(srcHsv.y, max(srcHsv.y, targetHsv.y * 0.8), strength);
    fragColor = vec4(hsv2rgb(vec3(hue, saturation, srcHsv.z)), texel.a);
}
