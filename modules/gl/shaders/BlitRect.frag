#version 460 core

uniform sampler2D tex;
uniform vec4 rect; // x, y, width, height in normalized device coordinates [-1, 1]

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Convert gl_FragCoord to normalized device coordinates
    vec2 ndc = gl_FragCoord.xy / vec2(textureSize(tex, 0)) * 2.0 - 1.0;

    // Check if fragment is within the specified rectangle
    vec2 rectMin = rect.xy;
    vec2 rectMax = rect.xy + rect.zw;

    if (ndc.x >= rectMin.x && ndc.x <= rectMax.x &&
        ndc.y >= rectMin.y && ndc.y <= rectMax.y) {
        fragColor = texture(tex, texCoord);
    } else {
        discard; // Don't draw outside the rectangle
    }
}
