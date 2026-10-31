#version 460 core

// Kinetic Emission Shader
// The ink a release lets go of, read by the density bridge as its alpha

uniform sampler2D tex0;  // Held release R
uniform sampler2D tex1;  // Reservoir R, read only with use_reservoir
uniform float gain;      // Emission gain
uniform bool use_reservoir;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float emission = texture(tex0, texCoord).r * gain;
    if (use_reservoir) emission *= texture(tex1, texCoord).r;
    fragColor = vec4(emission, 0.0, 0.0, 1.0);
}
