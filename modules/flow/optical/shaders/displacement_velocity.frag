#version 460 core

// A thresholded displacement (uv per frame) as the trail bridge's velocity: in pixels of the flow's
// size, times `scale` and `force`, its length to `power`, clamped. The numpy reference is
// lucas_kanade.shape_displacement

uniform sampler2D displacement;
uniform float scale;        // velocity per pixel of displacement
uniform vec2 force;         // strength x, y (negative inverts)
uniform float power;

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.0001

void main() {
    vec2 size = vec2(textureSize(displacement, 0));
    vec2 flow = texture(displacement, texCoord).xy * size * scale * force;

    float magnitude = length(flow);
    flow = (magnitude > TINY) ? (flow / magnitude) * pow(magnitude, power) : vec2(0.0);
    flow = clamp(flow, vec2(-1.0), vec2(1.0));

    fragColor = vec4(flow, 0.0, 1.0);
}
