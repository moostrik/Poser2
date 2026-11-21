#version 460 core

layout(binding = 0) uniform sampler2D src;      // Input mask texture (red channel)
uniform float radius;        // Erosion radius (typically 1.0 for multi-pass)
#define texelSize (1.0 / vec2(textureSize(src, 0)))

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float minVal = 1.0;

    // 3x3 erosion kernel - samples 9 pixels and takes minimum
    // Perfect for multiple passes to achieve larger erosion
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize * radius;
            minVal = min(minVal, texture(src, texCoord + offset).r);
        }
    }

    // Output eroded mask to red channel (grayscale mask format)
    fragColor = vec4(minVal, minVal, minVal, 1.0);
}
