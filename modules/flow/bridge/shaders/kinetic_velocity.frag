#version 460 core

// Kinetic Velocity Shader
// The velocity output: the fast trail, blending to the slow trail where the motion releases,
// so a stop leaves the fluid coasting instead of braked. The numpy reference is kinetics.drop

uniform sampler2D tex0;  // Fast trail RG
uniform sampler2D tex1;  // Slow trail RG
uniform sampler2D tex2;  // Acceleration R
uniform float drop;      // How far a full release blends to the slow trail, 0..1

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    vec2 fast = texture(tex0, texCoord).xy;
    vec2 slow = texture(tex1, texCoord).xy;
    float release = max(0.0, -texture(tex2, texCoord).r);

    // Current, not held: a new motion takes over the moment it starts
    float weight = clamp(release / max(length(slow), EPSILON), 0.0, 1.0);
    fragColor = vec4(mix(fast, slow, drop * weight), 0.0, 1.0);
}
