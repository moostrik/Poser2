#version 460 core

uniform sampler2D src;
uniform sampler2D dst;
uniform float amount;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = mix(texture(src, texCoord), texture(dst, texCoord), amount);
}
