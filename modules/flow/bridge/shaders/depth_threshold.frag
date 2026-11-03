#version 460 core

// Depth Threshold Shader
// Takes the noise floor off a signed field, continuously. The numpy reference is kinetics.depth_threshold

uniform sampler2D tex0;  // Motion toward the camera R
uniform float threshold; // Size taken off, in the field's units

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float value = texture(tex0, texCoord).r;
    fragColor = vec4(sign(value) * max(abs(value) - threshold, 0.0), 0.0, 0.0, 1.0);
}
