#version 460 core

// A mask's 0.5 threshold, its edge antialiased over the pixel the threshold crosses (`fwidth`).

layout(binding = 0) uniform sampler2D tex0;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float value = texture(tex0, texCoord).r;
    float aa = fwidth(value) * 0.5;
    float result = clamp(smoothstep(0.5 - aa, 0.5 + aa, value), 0.0, 1.0);
    fragColor = vec4(result, result, result, 1.0);
}
