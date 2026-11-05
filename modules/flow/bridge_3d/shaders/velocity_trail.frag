#version 460 core

// Velocity Trail Shader
// One unit-gain Ema step of the velocity trail bridge: weight * previous + (1 - weight) * input.
// x/y takes the attack weight where its speed rises above the trail's, else the release weight;
// z takes the release weight. The numpy reference is velocity_trail_bridge.velocity_trail

uniform sampler2D tex0;     // Previous trail RGB
uniform sampler2D tex1;     // Shaped velocity RGB
uniform float attack;       // What remains of the trail where the x/y speed rises
uniform float release;      // What remains elsewhere, and of z

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 previous = texture(tex0, texCoord).xyz;
    vec3 velocity = texture(tex1, texCoord).xyz;

    float weight = length(velocity.xy) > length(previous.xy) ? attack : release;
    vec2 xy = mix(velocity.xy, previous.xy, weight);
    float z = mix(velocity.z, previous.z, release);

    fragColor = vec4(xy, z, 0.0);   // w unused
}
