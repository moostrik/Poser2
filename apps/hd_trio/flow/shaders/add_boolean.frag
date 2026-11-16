#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec4 fragColor;

// Input textures
uniform sampler2D uBase;  // Base obstacle texture
uniform sampler2D uBlend; // Blend obstacle texture (will be rounded to 0 or 1)

void main() {
    vec2 st = texCoord;

    // Sample textures
    vec4 base = texture(uBase, st);
    vec4 blend = texture(uBlend, st);

    // Round blend values to 0 or 1 (boolean operation)
    blend = clamp(round(blend), 0.0, 1.0);

    // Add (union) operation: result is 1 if either is 1
    fragColor = base + blend;
}
