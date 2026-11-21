#version 460 core

layout(binding = 0) uniform sampler2D src;     // Input mask texture
uniform float threshold;     // Step threshold value (0.0 - 1.0)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float texel0 = texture(src, texCoord).r;
    float result = step(threshold, texel0);
    fragColor = vec4(result, result, result, 1.0);
}