#version 460 core

// Depth Velocity Shader
// Motion toward the camera in view widths per frame, from two smoothed depths: the change over the
// metres one view width spans at the depth, `depth × view_span`, so it is measured like the flow's
// x/y. 0 without depth at both, at a jump, and without a span. With `warp` the previous depth is the
// one where the pixel came from, `displacement` (uv per frame) back, taken nearest: the change is
// measured along the motion, so a surface sliding sideways over a slope reads no motion toward the
// camera. The numpy reference is kinetics.depth_displacement, with kinetics.warped

uniform sampler2D tex0;  // Smoothed depth R
uniform sampler2D tex1;  // Previous smoothed depth R
uniform sampler2D displacement;     // uv per frame (RG), with warp
uniform bool warp;
uniform float view_span; // Metres one view width spans per metre of depth
uniform float jump;      // More change than this is an edge (m)

in vec2 texCoord;
out vec4 fragColor;

float previous_at(vec2 uv) {
    if (!warp) {
        return texture(tex1, uv).r;
    }
    ivec2 size = textureSize(tex1, 0);
    ivec2 from = clamp(ivec2(floor((uv - texture(displacement, uv).xy) * vec2(size))), ivec2(0), size - 1);
    return texelFetch(tex1, from, 0).r;
}

void main() {
    float depth = texture(tex0, texCoord).r;
    float previous = previous_at(texCoord);
    float change = depth - previous;
    bool moving = depth > 0.0 && previous > 0.0 && abs(change) <= jump && view_span > 0.0;
    fragColor = vec4(moving ? -change / (depth * view_span) : 0.0, 0.0, 0.0, 1.0);
}
