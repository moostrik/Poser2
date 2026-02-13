#version 460 core

// Multiply Force Shader
// Ported from ofxFlowTools ftMultiplyForceShader.h
// Simple scalar multiplication for timestep scaling

uniform sampler2D src;
uniform float scale;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = texture(src, texCoord) * scale;
    fragColor = color;
}
