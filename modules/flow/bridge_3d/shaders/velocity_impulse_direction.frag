#version 460 core

// Velocity Impulse Direction Shader
// An impulse as a velocity: its amount along a trail's 3D direction. The numpy reference is
// velocity_impulse_bridge.impulse_velocity

uniform sampler2D tex0;  // Trail RGB: the direction
uniform sampler2D tex1;  // Impulse R: the amount

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;     // EPSILON in trails

void main() {
    vec3 trail = texture(tex0, texCoord).xyz;
    float amount = max(0.0, texture(tex1, texCoord).r);

    float trail_length = length(trail);
    vec3 unit = trail_length > EPSILON ? trail / trail_length : vec3(0.0);
    fragColor = vec4(unit * amount, 0.0);   // w unused
}
