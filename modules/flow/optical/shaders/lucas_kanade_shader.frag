#version 460 core

// Lucas–Kanade, one mip level: refines the displacement so far (uv per frame) by the shift that carries
// the previous frame, warped by it, onto the current one, solved over a (2·window+1)² window of this
// level's texels. Run from the coarsest level to the finest. The numpy reference is lucas_kanade_flow.lucas_kanade

uniform sampler2D previous;     // luminance, mipmapped
uniform sampler2D current;      // luminance, mipmapped
uniform sampler2D estimate;     // displacement so far, uv per frame (RG)
uniform int level;
uniform int window;
uniform float damping;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 d = texture(estimate, texCoord).xy;
    float lod = float(level);
    vec2 texel = 1.0 / vec2(textureSize(current, level));
    vec2 dx = vec2(texel.x, 0.0);
    vec2 dy = vec2(0.0, texel.y);

    float sxx = 0.0, sxy = 0.0, syy = 0.0, sxt = 0.0, syt = 0.0;
    for (int j = -window; j <= window; j++) {
        for (int i = -window; i <= window; i++) {
            vec2 p = texCoord + vec2(i, j) * texel;
            vec2 q = p - d;
            float gx = 0.25 * (textureLod(current, p + dx, lod).r - textureLod(current, p - dx, lod).r
                             + textureLod(previous, q + dx, lod).r - textureLod(previous, q - dx, lod).r);
            float gy = 0.25 * (textureLod(current, p + dy, lod).r - textureLod(current, p - dy, lod).r
                             + textureLod(previous, q + dy, lod).r - textureLod(previous, q - dy, lod).r);
            float gt = textureLod(current, p, lod).r - textureLod(previous, q, lod).r;
            sxx += gx * gx;
            sxy += gx * gy;
            syy += gy * gy;
            sxt += gx * gt;
            syt += gy * gt;
        }
    }
    sxx += damping;
    syy += damping;
    float det = sxx * syy - sxy * sxy;
    vec2 delta = abs(det) > 1e-12 ? -vec2(syy * sxt - sxy * syt, sxx * syt - sxy * sxt) / det : vec2(0.0);
    fragColor = vec4(d + delta * texel, 0.0, 1.0);
}
