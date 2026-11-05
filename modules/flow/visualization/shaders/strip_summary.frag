#version 460 core

// Strip Summary Shader
// A texture's channels as amounts, in one small texture whose top mip is their mean

uniform sampler2D tex0;  // The texture plotted

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = abs(texture(tex0, texCoord));
}
