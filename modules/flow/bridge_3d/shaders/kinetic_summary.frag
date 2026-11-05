#version 460 core

// Kinetic Summary Shader
// The fields the kinetic strip plots, in one small texture whose top mip is their mean

uniform sampler2D tex0;  // Fast trail RGB
uniform sampler2D tex1;  // Acceleration R
uniform sampler2D tex2;  // Emission R

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float speed = length(texture(tex0, texCoord).xyz);
    float acceleration = texture(tex1, texCoord).r;
    float emission = texture(tex2, texCoord).r;
    fragColor = vec4(speed, max(0.0, acceleration), max(0.0, -acceleration), emission);
}
