#version 460 core

// A matte from the color's brightness, its brightest channel (HSV's value): darkness becomes transparency. The
// alpha becomes 1 − amount + amount × that brightness, times the alpha it had; the color as it is.

uniform sampler2D src;
uniform float amount;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    float value = max(max(color.r, color.g), color.b);
    fragColor = vec4(color.rgb, color.a * (1.0 - amount + amount * clamp(value, 0.0, 1.0)));
}
