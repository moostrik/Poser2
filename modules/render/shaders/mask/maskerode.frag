#version 460 core

uniform sampler2D tex0;      // Input mask texture (red channel)
uniform float radius;        // Erosion radius (typically 1.0 for multi-pass)
uniform vec2 texelSize;      // 1.0 / texture_dimensions

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float minVal = 1.0;

    // 3x3 erosion kernel - samples 9 pixels and takes minimum
    // Perfect for multiple passes to achieve larger erosion
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize * radius;
            minVal = min(minVal, texture(tex0, texCoord + offset).r);
        }
    }

    // Output eroded mask to red channel (grayscale mask format)
    fragColor = vec4(minVal, minVal, minVal, 1.0);
}
