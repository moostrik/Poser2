#version 460 core

layout(binding = 0) uniform sampler2D src;
uniform mat3 white_balance;     // on linear sRGB: the chosen light's white to 6500 K's
uniform float exposure;
uniform float offset;
uniform float gamma;
uniform float contrast;
uniform float saturation;

in vec2 texCoord;
out vec4 fragColor;

#include "color.glsl"

void main() {
    vec4 color = texture(src, texCoord);
    vec3 balanced = linear_to_srgb(white_balance * srgb_to_linear(color.rgb));
    vec3 graded = balanced * exposure + offset;
    graded = pow(max(graded, 0.0), vec3(1.0 / gamma));
    graded = clamp((graded - 0.5) * contrast + 0.5, 0.0, 1.0);
    vec3 hsv = rgb2hsv(graded);
    hsv.y = clamp(hsv.y * saturation, 0.0, 1.0);
    fragColor = vec4(hsv2rgb(hsv), color.a);
}
