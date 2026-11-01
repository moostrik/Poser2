#version 460 core

// Depth Smooth Shader
// Depth through an Ema, because frame to frame it is mostly noise. Only the player counts: the
// source holds metres in red and the mask in alpha, and a few texels of mask must hold on each
// side, so a mask edge never reads as a distance in between. A pixel that jumps, as at a silhouette,
// restarts from the new depth. The numpy reference is kinetics.smooth_depth

uniform sampler2D tex0;  // Depth: metres R, mask A
uniform sampler2D tex1;  // Previous smoothed depth R
uniform float weight;    // Weight of the previous value
uniform float jump;      // More change than this per frame is an edge (m)

in vec2 texCoord;
out vec4 fragColor;

const float MARGIN = 3.0;   // texels

float depth_at(vec2 uv) {
    vec4 texel = texture(tex0, uv);
    return texel.a > 0.5 ? texel.r : 0.0;
}

void main() {
    vec2 step_uv = MARGIN / vec2(textureSize(tex1, 0));
    float depth = depth_at(texCoord);
    float nearest = min(min(depth_at(texCoord + vec2(step_uv.x, 0.0)), depth_at(texCoord - vec2(step_uv.x, 0.0))),
                        min(depth_at(texCoord + vec2(0.0, step_uv.y)), depth_at(texCoord - vec2(0.0, step_uv.y))));
    if (min(depth, nearest) <= 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float previous = texture(tex1, texCoord).r;
    bool tracked = previous > 0.0 && abs(depth - previous) <= jump;
    fragColor = vec4(tracked ? mix(depth, previous, weight) : depth, 0.0, 0.0, 1.0);
}
