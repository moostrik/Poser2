#version 460 core

uniform sampler2D correlationTexture;
uniform vec3 lineColor;
uniform float lineWidth;
uniform vec2 viewportSize;
uniform int pairIndex;
uniform int totalPairs;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float pairHeight = 1.0 / float(totalPairs);
    float pairY = float(pairIndex) * pairHeight;

    // Check if we're in the current pair's section
    if (texCoord.y < pairY || texCoord.y > pairY + pairHeight) {
        discard;
    }

    // Normalize Y coordinate within the pair section
    float localY = (texCoord.y - pairY) / pairHeight;

    // Sample the correlation data
    float correlation = texture(correlationTexture, vec2(texCoord.x, 0.0)).r;

    // Draw line based on correlation value
    float distance = abs(localY - correlation);
    float pixelLineWidth = lineWidth / viewportSize.y;
    float alpha = 1.0 - smoothstep(0.0, pixelLineWidth, distance);

    fragColor = vec4(lineColor, alpha);
}