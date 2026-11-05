#version 460 core

// Scale Velocity Shader
// The kinetic bridge's input: a velocity's x, y and z, each times its scale

uniform sampler2D tex0;     // Velocity RGB
uniform vec3 scale;         // Multiplier on x, y and z

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(texture(tex0, texCoord).xyz * scale, 0.0);   // w unused
}
