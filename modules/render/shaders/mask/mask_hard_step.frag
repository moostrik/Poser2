#version 460 core

uniform sampler2D tex0;      // Input mask texture
uniform float threshold;     // Step threshold value (0.0 - 1.0)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float texel0 = texture(tex0, texCoord).r;
    float result = step(threshold, texel0);
    fragColor = vec4(result, result, result, 1.0);
}