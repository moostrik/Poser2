#version 460 core

layout(location = 0) in vec2 position;
out vec2 texCoord;

uniform vec2 resolution;
uniform bool flipX = false;
uniform bool flipY = false;

void main() {
    // Normalize pixel coordinates to [0, 1]
    vec2 normalized = position / resolution;

    // Calculate texture coordinates with optional flipping
    texCoord = normalized;
    if (flipX) texCoord.x = 1.0 - texCoord.x;
    if (flipY) texCoord.y = 1.0 - texCoord.y;

    // Convert to NDC [-1, 1] with Y-flip for top-left origin
    vec2 ndc;
    ndc.x = normalized.x * 2.0 - 1.0;    // 0→-1, width→1
    ndc.y = 1.0 - normalized.y * 2.0;    // 0→1, height→-1 (FLIPPED)

    gl_Position = vec4(ndc, 0.0, 1.0);
}