#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec4 fragColor;

// Input textures
uniform sampler2D uDensity;     // Density field (RGBA16F) - each channel is a different track

// Track colors - up to 4 tracks mapped to RGBA channels
uniform vec4 uColors[4];

void main() {
    vec4 den = texture(uDensity, texCoord);

    // Map each density channel to its corresponding track color
    // R channel -> track 0, G -> track 1, B -> track 2, A -> track 3
    vec3 result = vec3(0.0);
    result += den.r * uColors[0].rgb;
    result += den.g * uColors[1].rgb;
    result += den.b * uColors[2].rgb;
    result += den.a * uColors[3].rgb;

    // Alpha from max of densities (for transparency blending)
    float alpha = max(max(den.r, den.g), max(den.b, den.a));

    fragColor = vec4(result, alpha);
}
