#version 460 core

// Push-pull (Gortler et al. 1996), one of its passes by `stage`, on pairs (depth × weight, weight) in red and green:
//   0  push: a level down, each pixel the mean of the up to four it covers; from the read (`first`), a pair is its
//      depth with weight 1 where there is one, (0, 0) where not
//   1  pull: a level's pair plus (1 − its weight) times the coarser pulled level's, read bilinearly
//   2  the last pull, onto the read: a pixel with a depth keeps it and its confidence (the read's green), one
//      without takes the pulled level's depth (its pair's ratio) and a confidence of 0; 0 and 0 where the mask is 0
//      or nothing was found
// A coarser pixel's center lies in the middle of the two it covers, its edges clamped; rows count from the bottom,
// as GL holds a texture.

layout(binding = 0) uniform sampler2D tex0;     // 0: the finer level (or the read); 1: this level's push; 2: the read (meters, confidence)
layout(binding = 1) uniform sampler2D tex1;     // 1, 2: the coarser pulled level
layout(binding = 2) uniform sampler2D tex2;    // 2: the mask (red, 0 to 1)
uniform int stage;
uniform bool first;         // 0: tex0 is the read, meters in red

out vec4 fragColor;

vec2 pair(ivec2 p) {
    vec4 value = texelFetch(tex0, p, 0);
    if (first) return value.r > 0.0 ? vec2(value.r, 1.0) : vec2(0.0);
    return value.rg;
}

vec2 coarse(ivec2 p) {
    ivec2 size = textureSize(tex1, 0);
    vec2 at = (vec2(p) + 0.5) * 0.5 - 0.5;
    ivec2 i = ivec2(floor(at));
    vec2 f = at - vec2(i);
    ivec2 a = clamp(i, ivec2(0), size - 1);
    ivec2 b = clamp(i + 1, ivec2(0), size - 1);
    vec2 bottom = mix(texelFetch(tex1, ivec2(a.x, a.y), 0).rg, texelFetch(tex1, ivec2(b.x, a.y), 0).rg, f.x);
    vec2 top = mix(texelFetch(tex1, ivec2(a.x, b.y), 0).rg, texelFetch(tex1, ivec2(b.x, b.y), 0).rg, f.x);
    return mix(bottom, top, f.y);
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    if (stage == 0) {
        ivec2 size = textureSize(tex0, 0);
        vec2 sum = vec2(0.0);
        float count = 0.0;
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                ivec2 at = p * 2 + ivec2(x, y);
                if (at.x < size.x && at.y < size.y) {
                    sum += pair(at);
                    count += 1.0;
                }
            }
        }
        fragColor = vec4(sum / count, 0.0, 0.0);
    } else if (stage == 1) {
        vec2 level = texelFetch(tex0, p, 0).rg;
        fragColor = vec4(level + (1.0 - level.g) * coarse(p), 0.0, 0.0);
    } else {
        vec2 read = texelFetch(tex0, p, 0).rg;
        vec2 pulled = coarse(p);
        vec2 filled = read.r > 0.0 ? read : vec2(pulled.g > 0.0 ? pulled.r / pulled.g : 0.0, 0.0);
        fragColor = vec4(texelFetch(tex2, p, 0).r > 0.0 ? filled : vec2(0.0), 0.0, 1.0);
    }
}
