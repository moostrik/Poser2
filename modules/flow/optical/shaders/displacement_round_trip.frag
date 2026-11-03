#version 460 core

// Displacement Round Trip Shader
// The forward displacement (uv per frame) where it comes back: the backward one, read where the pixel
// was in the previous frame, cancels it to within `largest_miss` pixels. Where it does not, as where a surface is
// covered or uncovered, there is no displacement. The numpy reference is
// lucas_kanade_flow.round_trip_displacement

uniform sampler2D forward;
uniform sampler2D backward;
uniform float largest_miss;  // px

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 size = vec2(textureSize(forward, 0));
    vec2 there = texture(forward, texCoord).xy;
    vec2 back = texture(backward, texCoord - there).xy;     // where the pixel was in the previous frame
    bool returns = length((there + back) * size) <= largest_miss;
    fragColor = vec4(returns ? there : vec2(0.0), 0.0, 1.0);
}
