#version 460 core

// The 3 x 3 median by McGuire's exchange network ("A Fast, Small-Radius GPU Median Filter", ShaderX6)

layout(binding = 0) uniform sampler2D src;

in vec2 texCoord;
out vec4 fragColor;

#define s2(a, b)            { vec4 t = a; a = min(t, b); b = max(t, b); }
#define mn3(a, b, c)        s2(a, b); s2(a, c);
#define mx3(a, b, c)        s2(b, c); s2(a, c);
#define mnmx3(a, b, c)      mx3(a, b, c); s2(a, b);
#define mnmx4(a, b, c, d)   s2(a, b); s2(c, d); s2(a, c); s2(b, d);
#define mnmx5(a, b, c, d, e) s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f);

void main() {
    ivec2 last = textureSize(src, 0) - 1;
    ivec2 center = ivec2(texCoord * vec2(last + 1));
    vec4 v[9];
    int i = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            v[i++] = texelFetch(src, clamp(center + ivec2(x, y), ivec2(0), last), 0);
        }
    }
    mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
    mnmx5(v[1], v[2], v[3], v[4], v[6]);
    mnmx4(v[2], v[3], v[4], v[7]);
    mnmx3(v[3], v[4], v[8]);
    fragColor = v[4];
}
