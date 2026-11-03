#version 460 core

// Displacement Median Shader
// Each component of the displacement (uv per frame) as the median of its 5×5 samples, `spacing`
// (uv) apart, nearest: one level's outliers go before the next level refines on them. The numpy
// reference is lucas_kanade_flow.median_displacement

uniform sampler2D displacement;
uniform vec2 spacing;   // uv between two samples

in vec2 texCoord;
out vec4 fragColor;

float median25(float values[25]) {
    for (int i = 1; i < 25; i++) {          // insertion sort; 25 values
        float value = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > value) {
            values[j + 1] = values[j];
            j--;
        }
        values[j + 1] = value;
    }
    return values[12];
}

void main() {
    ivec2 size = textureSize(displacement, 0);
    float xs[25];
    float ys[25];
    int n = 0;
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            vec2 uv = texCoord + vec2(i, j) * spacing;
            ivec2 texel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
            vec2 d = texelFetch(displacement, texel, 0).xy;
            xs[n] = d.x;
            ys[n] = d.y;
            n++;
        }
    }
    fragColor = vec4(median25(xs), median25(ys), 0.0, 1.0);
}
