#version 460 core

// One Euro Trail Shader
// One step of the velocity trail as a 1€ filter: x/y low-passed with a cutoff that grows with its rate of
// change, cutoff = min_cutoff + beta * rate, so a still or steady motion is smoothed long and a changing
// one follows at once; z low-passed by `release`. The numpy reference is velocity_trail_bridge.one_euro_trail

uniform sampler2D tex0;     // Previous trail RGB
uniform sampler2D tex1;     // Velocity RGB
uniform sampler2D tex2;     // Rate R, widths/s²
uniform float min_cutoff;   // Cutoff at rest, Hz
uniform float beta;         // Cutoff added per widths/s² of rate, Hz
uniform float dt;           // Seconds this step
uniform float release;      // What remains of z

in vec2 texCoord;
out vec4 fragColor;

#define TAU 6.28318530718

void main() {
    vec3 previous = texture(tex0, texCoord).xyz;
    vec3 velocity = texture(tex1, texCoord).xyz;
    float cutoff = min_cutoff + beta * texture(tex2, texCoord).r;
    float weight = exp(-dt * TAU * cutoff);

    vec2 xy = mix(velocity.xy, previous.xy, weight);
    float z = mix(velocity.z, previous.z, release);
    fragColor = vec4(xy, z, 1.0);
}
