#version 460 core

// Range flow, one sampling stride: refines the displacement so far and measures the depth change W by
// least squares on Zx·u + Zy·v − W = −Zt over a (2·window+1)² window of samples `stride` texels apart.
// A sample counts when it, its gradient neighbours and its previous depth (where it came from) all have
// depth within `jump` of each other and of the center pixel's: a window never mixes two surfaces. Run
// from the widest stride to 1. The numpy reference is range_flow.range_flow

uniform sampler2D previous;     // smoothed depth in metres (R), 0 for none
uniform sampler2D current;      // smoothed depth in metres (R), 0 for none
uniform sampler2D estimate;     // displacement so far in uv (RG); toward the camera, m/s (B); share counted (A)
uniform int stride;
uniform int window;
uniform float regularizer;
uniform float jump;
uniform float dt;

in vec2 texCoord;
out vec4 fragColor;

ivec2 size;

float at(sampler2D depth, vec2 pixel) {
    return texelFetch(depth, clamp(ivec2(floor(pixel + 0.5)), ivec2(0), size - 1), 0).r;
}

void main() {
    size = textureSize(current, 0);
    vec2 pixel = gl_FragCoord.xy - 0.5;
    vec4 last = texture(estimate, texCoord);
    vec2 d = last.xy * vec2(size);                  // pixels per frame
    float center = at(current, pixel);
    float s = float(stride);

    mat3 m = mat3(0.0);
    vec3 b = vec3(0.0);
    float count = 0.0;
    float total = 0.0;
    for (int j = -window; j <= window; j++) {
        for (int i = -window; i <= window; i++) {
            total += 1.0;
            vec2 p = pixel + vec2(i, j) * s;
            float z1 = at(current, p);
            float z0 = at(previous, p - d);
            float right = at(current, p + vec2(s, 0.0));
            float left = at(current, p - vec2(s, 0.0));
            float down = at(current, p + vec2(0.0, s));
            float up = at(current, p - vec2(0.0, s));
            bool valid = z1 > 0.0 && z0 > 0.0 && right > 0.0 && left > 0.0 && down > 0.0 && up > 0.0 && center > 0.0
                && abs(z1 - center) <= jump && abs(z1 - z0) <= jump
                && abs(right - z1) <= jump && abs(left - z1) <= jump && abs(down - z1) <= jump && abs(up - z1) <= jump;
            if (!valid) {
                continue;
            }
            vec3 row = vec3((right - left) / (2.0 * s), (down - up) / (2.0 * s), -1.0);
            float gt = z1 - z0;
            m += outerProduct(row, row);
            b += row * -gt;
            count += 1.0;
        }
    }

    if (count < 6.0) {
        fragColor = vec4(last.xy, 0.0, 0.0);
        return;
    }
    m[0][0] += regularizer;
    m[1][1] += regularizer;
    m[2][2] += 1e-6;
    if (abs(determinant(m)) < 1e-12) {
        fragColor = vec4(last.xy, 0.0, 0.0);
        return;
    }
    vec3 solution = inverse(m) * b;
    vec2 displacement = (d + solution.xy) / vec2(size);
    fragColor = vec4(displacement, -solution.z / dt, count / total);
}
