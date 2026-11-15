#version 460 core

uniform sampler2D src;
uniform vec4 color;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float mask = texture(src, texCoord).r;
    fragColor = vec4(color.rgb, mask);
}
