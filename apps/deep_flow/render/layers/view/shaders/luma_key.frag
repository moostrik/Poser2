#version 460 core

// The luma key: darkness becomes transparency. The alpha becomes 1 − key + key × the color's brightness, its
// brightest channel (HSV's value), times the alpha it had; the RGB as it is.

uniform sampler2D src;
uniform float key;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    float value = max(max(color.r, color.g), color.b);
    fragColor = vec4(color.rgb, color.a * (1.0 - key + key * clamp(value, 0.0, 1.0)));
}
