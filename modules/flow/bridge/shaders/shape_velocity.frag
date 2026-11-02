#version 460 core

// Shape Velocity Shader
// The trail bridge's input: the x/y length to `power`, then clamped to ±1 per axis; z passes as it
// is. The numpy reference is smooth_trail.shape_velocity

uniform sampler2D tex0;     // Velocity RGB
uniform float power;        // Power curve on the x/y length

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.0001

void main() {
    vec3 velocity = texture(tex0, texCoord).xyz;
    vec2 flow = velocity.xy;

    float magnitude = length(flow);
    flow = (magnitude > TINY) ? (flow / magnitude) * pow(magnitude, power) : vec2(0.0);
    flow = clamp(flow, vec2(-1.0), vec2(1.0));

    fragColor = vec4(flow, velocity.z, 1.0);
}
