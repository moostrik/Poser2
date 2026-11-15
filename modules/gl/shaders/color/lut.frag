#version 460 core

uniform sampler2D src;
uniform sampler3D lut;
uniform float strength;
uniform vec3 domainMin;
uniform vec3 domainMax;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 originalColor = texture(src, texCoord);

    // Normalize input color to LUT domain [0, 1]
    vec3 domainRange = domainMax - domainMin;
    vec3 normalizedColor = (originalColor.rgb - domainMin) / domainRange;

    // Clamp to valid range for LUT lookup
    normalizedColor = clamp(normalizedColor, 0.0, 1.0);

    // Sample the 3D LUT
    vec3 lutColor = texture(lut, normalizedColor).rgb;

    // Blend between original and LUT result based on strength
    vec3 finalColor = mix(originalColor.rgb, lutColor, strength);

    fragColor = vec4(finalColor, originalColor.a);
}
