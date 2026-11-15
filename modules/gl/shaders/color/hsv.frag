#version 460 core

#include "color.glsl"

uniform sampler2D src;
uniform float hue;
uniform float saturation;
uniform float value;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    vec3 hsv = rgb2hsv(color.rgb);
    hsv.x = fract(hsv.x + hue);                     // the hue wraps
    hsv.y = clamp(hsv.y * saturation, 0.0, 1.0);    // above 1 a saturation gives negative colors
    hsv.z = clamp(hsv.z * value, 0.0, 1.0);
    fragColor = vec4(hsv2rgb(hsv), color.a);
}
