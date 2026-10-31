#version 460 core

// Kinetic Emission Shader
// The ink a release lets go of, read by the density bridge as its alpha

uniform sampler2D tex0;  // Held release R
uniform float gain;      // Emission gain

in vec2 texCoord;
out vec4 fragColor;

void main() {
    fragColor = vec4(texture(tex0, texCoord).r * gain, 0.0, 0.0, 1.0);
}
