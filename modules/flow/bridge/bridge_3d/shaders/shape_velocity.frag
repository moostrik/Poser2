#version 460 core

// Shape Velocity Shader
// The velocity trail bridge's input: x/y as it is, z rolled off softly toward `z_limit`
// (z_limit * tanh(z / z_limit)), linear below it

uniform sampler2D tex0;     // Velocity RGB, widths/s
uniform float z_limit;      // The z the roll-off approaches

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 velocity = texture(tex0, texCoord).xyz;
    float z = z_limit * tanh(velocity.z / z_limit);

    fragColor = vec4(velocity.xy, z, 1.0);
}
