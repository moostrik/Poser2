#version 460 core

// The noise floor off a displacement (uv per frame): its length in pixels loses `threshold`, and
// below it is 0. Below the threshold is the solve's jitter on a still surface, not motion. The
// direction is kept, and what is above stays continuous. The numpy reference is
// lucas_kanade_flow.threshold_displacement

uniform sampler2D displacement;
uniform float threshold;    // px per frame

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 size = vec2(textureSize(displacement, 0));
    vec2 uv = texture(displacement, texCoord).xy;
    float pixels = length(uv * size);
    float kept = max(pixels - threshold, 0.0);
    fragColor = vec4(pixels > 1e-6 ? uv * (kept / pixels) : vec2(0.0), 0.0, 1.0);
}
