#version 460 core

// Signed Map Shader
// Draws a scalar field in two colors: positive warm, negative cold

uniform sampler2D tex0;  // Scalar field in R
uniform float scale;     // Multiplier on the value

in vec2 texCoord;
out vec4 fragColor;

const vec3 POSITIVE = vec3(1.0, 0.45, 0.1);
const vec3 NEGATIVE = vec3(0.1, 0.5, 1.0);

void main() {
    float value = texture(tex0, texCoord).r * scale;
    fragColor = vec4((value >= 0.0 ? POSITIVE : NEGATIVE) * min(abs(value), 1.0), 1.0);   // clamped, so the hue survives
}
