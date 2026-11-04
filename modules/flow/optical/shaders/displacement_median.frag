#version 460 core

// Displacement Median Shader
// Each component of the displacement (uv per frame) as the median of its (2·radius+1)² samples,
// `spacing` (uv) apart, nearest: one level's outliers go before the next level refines on them. The
// numpy reference is lucas_kanade_flow.median_displacement

uniform sampler2D displacement;
uniform vec2 spacing;   // uv between two samples
uniform int radius;     // 1 = 3×3, 2 = 5×5

in vec2 texCoord;
out vec4 fragColor;

float median(float values[25], int n) {
    for (int i = 1; i < n; i++) {           // insertion sort; up to 25 values
        float value = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > value) {
            values[j + 1] = values[j];
            j--;
        }
        values[j + 1] = value;
    }
    return values[n / 2];
}

void main() {
    ivec2 size = textureSize(displacement, 0);
    float xs[25];
    float ys[25];
    int n = 0;
    for (int j = -radius; j <= radius; j++) {
        for (int i = -radius; i <= radius; i++) {
            vec2 uv = texCoord + vec2(i, j) * spacing;
            ivec2 texel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
            vec2 d = texelFetch(displacement, texel, 0).xy;
            xs[n] = d.x;
            ys[n] = d.y;
            n++;
        }
    }
    fragColor = vec4(median(xs, n), median(ys, n), 0.0, 1.0);
}
