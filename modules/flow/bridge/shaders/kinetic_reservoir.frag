#version 460 core

// Kinetic Reservoir Shader
// Ink stored per pixel: motion fills it, it leaks, and what a release emits leaves it.
// The numpy reference is kinetics.fill_reservoir

uniform sampler2D tex0;  // Previous reservoir R
uniform sampler2D tex1;  // Fast trail RGB
uniform sampler2D tex2;  // Held release R
uniform float leak;      // What remains after the leak this update
uniform float dt;        // Seconds this update
uniform float gain;      // Emission gain

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float reservoir = texture(tex0, texCoord).r * leak + length(texture(tex1, texCoord).xyz) * dt;
    float emitted = texture(tex2, texCoord).r * gain * dt;
    fragColor = vec4(reservoir * max(0.0, 1.0 - emitted), 0.0, 0.0, 1.0);
}
