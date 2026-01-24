#version 460 core

uniform sampler2D tex;
uniform vec4 tint;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = texture(tex, texCoord) * tint;
}
