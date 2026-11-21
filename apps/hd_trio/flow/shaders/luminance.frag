#version 460 core

#include "color.glsl"

layout(binding = 0) uniform sampler2D tex0;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(tex0, texCoord);
    fragColor = vec4(luminance(color.rgb) * color.a, 0.0, 0.0, 0.0);
}
