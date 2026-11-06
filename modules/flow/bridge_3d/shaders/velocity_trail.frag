#version 460 core

// Velocity Trail Shader
// One unit-gain Ema step of the velocity trail bridge: weight * previous + (1 - weight) * input. x, y and z
// alike take the attack weight where the 3D speed rises above the trail's, else the release weight; z's
// noise is the motion's to filter. The numpy reference is velocity_trail_bridge.velocity_trail

uniform sampler2D tex0;     // Previous trail RGB
uniform sampler2D tex1;     // Velocity RGB
uniform float attack;       // What remains of the trail where the speed rises
uniform float release;      // What remains elsewhere

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec3 previous = texture(tex0, texCoord).xyz;
    vec3 velocity = texture(tex1, texCoord).xyz;

    float weight = length(velocity) > length(previous) ? attack : release;
    fragColor = vec4(mix(velocity, previous, weight), 0.0);   // w unused
}
