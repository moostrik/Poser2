#version 460 core

// Shape Velocity Shader
// The trail bridge's input: the x/y length to `power`, then rolled off softly toward `limit`
// (limit * tanh(length / limit), so the direction holds); z passes as it is

uniform sampler2D tex0;     // Velocity RGB
uniform float power;        // Power curve on the x/y length
uniform float limit;        // The x/y length the roll-off approaches

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.0001

void main() {
    vec3 velocity = texture(tex0, texCoord).xyz;
    vec2 flow = velocity.xy;

    float magnitude = length(flow);
    float shaped = limit * tanh(pow(magnitude, power) / limit);
    flow = (magnitude > TINY) ? (flow / magnitude) * shaped : vec2(0.0);

    fragColor = vec4(flow, velocity.z, 1.0);
}
