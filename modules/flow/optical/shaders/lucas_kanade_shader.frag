#version 460 core

// Lucas–Kanade, one mip level: refines the displacement so far (uv per frame) by the shift that carries
// the previous frame, warped by it, onto the current one, solved over a (2·window+1)² window of this
// level's texels. Run from the coarsest level to the finest. Each sample is weighted: with `gaussian`, by its
// distance from the centre (sigma = window / 1.5); with `same_surface`, 0 where its depth is further than
// `tolerance` from the centre's (so the background does not drag a silhouette along, at every level);
// `damping` scales with the weight kept. The numpy reference is lucas_kanade_flow.lucas_kanade

uniform sampler2D previous;     // luminance, mipmapped
uniform sampler2D current;      // luminance, mipmapped
uniform sampler2D estimate;     // displacement so far, uv per frame (RG)
uniform int level;
uniform int window;
uniform float damping;
uniform sampler2D depth;        // metres (R), 0 where there is none, with same_surface
uniform bool same_surface;
uniform float tolerance;        // m
uniform bool gaussian;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 d = texture(estimate, texCoord).xy;
    float lod = float(level);
    vec2 texel = 1.0 / vec2(textureSize(current, level));
    vec2 dx = vec2(texel.x, 0.0);
    vec2 dy = vec2(0.0, texel.y);

    ivec2 depth_size = textureSize(depth, 0);
    float center = same_surface ? texelFetch(depth, clamp(ivec2(floor(texCoord * vec2(depth_size))), ivec2(0), depth_size - 1), 0).r : 0.0;
    float sigma = max(float(window), 1.0) / 1.5;

    float sxx = 0.0, sxy = 0.0, syy = 0.0, sxt = 0.0, syt = 0.0;
    float total = 0.0;
    float count = 0.0;
    for (int j = -window; j <= window; j++) {
        for (int i = -window; i <= window; i++) {
            vec2 p = texCoord + vec2(i, j) * texel;
            float w = gaussian ? exp(-float(i * i + j * j) / (2.0 * sigma * sigma)) : 1.0;
            if (center > 0.0) {
                float sample_depth = texelFetch(depth, clamp(ivec2(floor(p * vec2(depth_size))), ivec2(0), depth_size - 1), 0).r;
                w *= (sample_depth > 0.0 && abs(sample_depth - center) <= tolerance) ? 1.0 : 0.0;
            }
            total += w;
            count += 1.0;
            vec2 q = p - d;
            float gx = 0.25 * (textureLod(current, p + dx, lod).r - textureLod(current, p - dx, lod).r
                             + textureLod(previous, q + dx, lod).r - textureLod(previous, q - dx, lod).r);
            float gy = 0.25 * (textureLod(current, p + dy, lod).r - textureLod(current, p - dy, lod).r
                             + textureLod(previous, q + dy, lod).r - textureLod(previous, q - dy, lod).r);
            float gt = textureLod(current, p, lod).r - textureLod(previous, q, lod).r;
            sxx += w * gx * gx;
            sxy += w * gx * gy;
            syy += w * gy * gy;
            sxt += w * gx * gt;
            syt += w * gy * gt;
        }
    }
    float kept_damping = damping * total / count;
    sxx += kept_damping;
    syy += kept_damping;
    float det = sxx * syy - sxy * sxy;
    vec2 delta = abs(det) > 1e-12 ? -vec2(syy * sxt - sxy * syt, sxx * syt - sxy * sxt) / det : vec2(0.0);
    fragColor = vec4(d + delta * texel, 0.0, 1.0);
}
