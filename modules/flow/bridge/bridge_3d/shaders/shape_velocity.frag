#version 460 core

// Shape Velocity Shader
// The velocity trail bridge's input: the x/y speed s bent around `pivot` to s * (s / pivot)^(curve - 1),
// then rolled off softly toward `limit` (limit * tanh(s / limit); 0 = none), so the direction holds;
// z rolled off toward `z_limit`, linear below it

uniform sampler2D tex0;     // Velocity RGB, widths/s
uniform float pivot;        // x/y speed the curve leaves as it is
uniform float curve;        // Exponent on the x/y speed around the pivot
uniform float limit;        // The x/y speed the roll-off approaches; 0 = no roll-off
uniform float z_limit;      // The z the roll-off approaches

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.0001

void main() {
    vec3 velocity = texture(tex0, texCoord).xyz;
    vec2 flow = velocity.xy;

    float speed = length(flow);
    float shaped = speed * pow(max(speed, TINY) / pivot, curve - 1.0);
    if (limit > 0.0) {
        shaped = limit * tanh(shaped / limit);
    }
    flow = (speed > TINY) ? (flow / speed) * shaped : vec2(0.0);
    float z = z_limit * tanh(velocity.z / z_limit);

    fragColor = vec4(flow, z, 1.0);
}
