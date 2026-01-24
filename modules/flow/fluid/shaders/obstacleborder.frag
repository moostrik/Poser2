#version 460 core

// Creates obstacle border mask
// White (1.0) at edges within border pixels, black (0.0) inside

in vec2 texCoord;
out vec4 fragColor;

uniform vec2 uResolution;  // Texture size in pixels
uniform float uBorder;     // Border width in pixels

void main() {
    // Convert to pixel coordinates
    vec2 pixel = texCoord * uResolution;

    // Check if within border region
    bool inBorder = pixel.x < uBorder ||
                    pixel.y < uBorder ||
                    pixel.x >= uResolution.x - uBorder ||
                    pixel.y >= uResolution.y - uBorder;

    fragColor = vec4(inBorder ? 1.0 : 0.0);
}
