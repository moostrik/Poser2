#version 460 core

// Z Position Resolve
// A z position as a monitor draws it: red the z position (red over green, 0 front to 1 back), green its
// weight; 0 where it has none

uniform sampler2D tex0;     // a z position: red the z position times its weight, green the weight

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 z_position = texture(tex0, texCoord).rg;
    fragColor = vec4(z_position.g > 1e-6 ? z_position.r / z_position.g : 0.0, z_position.g, 0.0, 0.0);
}
