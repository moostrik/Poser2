#version 460 core

// Clamp Shader
// Clamps all RGBA channels to a min/max range

uniform sampler2D src;
uniform float minVal;
uniform float maxVal;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord);
    fragColor = clamp(color, vec4(minVal), vec4(maxVal));
}
