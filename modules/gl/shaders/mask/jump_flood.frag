#version 460 core

// Jump flooding (Rong and Tan, "Jump Flooding in GPU with Applications to Voronoi Diagram and Distance Transform",
// I3D 2006). The map holds each pixel's nearest seed in texel coordinates, -1 for none.

layout(binding = 0) uniform sampler2D src;     // seed: the mask; step and distance: the map (RG32F)
uniform int stage;          // 0 seed, 1 step, 2 distance
uniform float threshold;    // seed: the mask's red that makes a seed
uniform int jump;           // step: pixels to look away

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 size = textureSize(src, 0);
    ivec2 p = ivec2(texCoord * vec2(size));
    if (stage == 0) {
        bool seed = texelFetch(src, p, 0).r >= threshold;
        fragColor = vec4(seed ? vec2(p) : vec2(-1.0), 0.0, 1.0);
    } else if (stage == 1) {
        vec2 best = texelFetch(src, p, 0).rg;
        float bestDistance = best.x < 0.0 ? 1e30 : dot(best - vec2(p), best - vec2(p));
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                ivec2 q = p + ivec2(x, y) * jump;
                if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) continue;
                vec2 seed = texelFetch(src, q, 0).rg;
                if (seed.x < 0.0) continue;
                float d = dot(seed - vec2(p), seed - vec2(p));
                if (d < bestDistance) {
                    bestDistance = d;
                    best = seed;
                }
            }
        }
        fragColor = vec4(best, 0.0, 1.0);
    } else {
        vec2 seed = texelFetch(src, p, 0).rg;
        fragColor = vec4(seed.x < 0.0 ? -1.0 : distance(seed, vec2(p)), 0.0, 0.0, 1.0);
    }
}
