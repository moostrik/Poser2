#version 460 core

// One Euro Rate Shader
// The 1€ filter's rate of change: how fast the x/y velocity changes, |input - trail| / dt in widths/s²,
// smoothed by its own low-pass. The numpy reference is velocity_trail_bridge.one_euro_rate

uniform sampler2D tex0;     // Previous trail RGB
uniform sampler2D tex1;     // Velocity RGB
uniform sampler2D tex2;     // Previous rate R
uniform float dt;           // Seconds this step
uniform float weight;       // What remains of the previous rate

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 previous = texture(tex0, texCoord).xy;
    vec2 velocity = texture(tex1, texCoord).xy;
    float rate = length(velocity - previous) / max(dt, 1e-6);
    fragColor = vec4(mix(rate, texture(tex2, texCoord).r, weight), 0.0, 0.0, 1.0);
}
