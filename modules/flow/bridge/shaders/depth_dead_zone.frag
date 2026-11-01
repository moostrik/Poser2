#version 460 core

// Depth Dead Zone Shader
// Takes the noise floor off a signed field, continuously. The numpy reference is kinetics.dead_zone

uniform sampler2D tex0;  // Velocity toward the camera R
uniform float width;     // Size taken off, in the field's units

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float value = texture(tex0, texCoord).r;
    fragColor = vec4(sign(value) * max(abs(value) - width, 0.0), 0.0, 0.0, 1.0);
}
