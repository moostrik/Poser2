#version 460 core

// Signed Map Shader
// Draws a scalar field in two colors: positive warm, negative cold;
// a depth field red toward the camera, green away

uniform sampler2D tex0;  // Scalar field in `channel`
uniform float scale;     // Multiplier on the value
uniform bool depth;      // Red and green
uniform int channel;     // 0 for red

in vec2 texCoord;
out vec4 fragColor;

const vec3 POSITIVE = vec3(1.0, 0.45, 0.1);
const vec3 NEGATIVE = vec3(0.1, 0.5, 1.0);
const vec3 TOWARD = vec3(1.0, 0.0, 0.0);
const vec3 AWAY = vec3(0.0, 1.0, 0.0);

void main() {
    float value = texture(tex0, texCoord)[channel] * scale;
    vec3 color = value >= 0.0 ? (depth ? TOWARD : POSITIVE) : (depth ? AWAY : NEGATIVE);
    fragColor = vec4(color * min(abs(value), 1.0), 1.0);   // clamped, so the hue survives
}
