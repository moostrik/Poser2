#version 460 core

// Kinetic Onset Shader
// The push of a starting motion: max(0, acceleration) along the fast trail's 3D direction

uniform sampler2D tex0;  // Fast trail RGB
uniform sampler2D tex1;  // Acceleration R
uniform float strength;  // Onset force

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    vec3 fast = texture(tex0, texCoord).xyz;
    float onset = max(0.0, texture(tex1, texCoord).r);

    float fast_length = length(fast);
    vec3 unit = fast_length > EPSILON ? fast / fast_length : vec3(0.0);
    fragColor = vec4(unit * onset * strength, 1.0);
}
