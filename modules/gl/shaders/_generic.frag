#version 460 core

uniform sampler2D tex0;

in vec2 texCoord;  // Normalized [0,1], (0,0) = top-left
out vec4 fragColor;

void main() {
    fragColor = texture(tex0, texCoord);
}