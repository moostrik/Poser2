#version 460 core

// Kinetic Velocity Shader
// The velocity output: the fast trail, blending to the slow trail where the motion releases,
// so a stop leaves the fluid coasting instead of braked. The numpy reference is kinetics.drop

uniform sampler2D tex0;  // Fast trail RGB
uniform sampler2D tex1;  // Slow trail RGB
uniform sampler2D tex2;  // Acceleration R
uniform float drop;      // How far a full release blends to the slow trail, 0..1
uniform bool depth;      // Write its z to R instead of its xy to RG

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    vec3 fast = texture(tex0, texCoord).xyz;
    vec3 slow = texture(tex1, texCoord).xyz;
    float release = max(0.0, -texture(tex2, texCoord).r);

    // Current, not held: a new motion takes over the moment it starts
    float weight = clamp(release / max(length(slow), EPSILON), 0.0, 1.0);
    vec3 velocity = mix(fast, slow, drop * weight);
    fragColor = depth ? vec4(velocity.z, 0.0, 0.0, 1.0) : vec4(velocity.xy, 0.0, 1.0);
}
