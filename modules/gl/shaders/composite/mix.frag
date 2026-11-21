#version 460 core

layout(binding = 0) uniform sampler2D src;
layout(binding = 1) uniform sampler2D dst;
uniform float amount;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = mix(texture(src, texCoord), texture(dst, texCoord), amount);
}
