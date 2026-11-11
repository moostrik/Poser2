#version 460 core

// Guided Filter (He, Sun and Tang 2010), one of its passes by `stage`, for a mask guided by a grayscale image:
//   0  the products: the guide's luminance I and the mask p as (I, p, I·I, I·p)
//   1  a box mean along `direction` over `radius` pixels, every channel
//   2  the coefficients from the means (mean I, mean p, mean I·I, mean I·p): a = cov(I, p) / (var(I) + ε),
//      b = mean p − a · mean I
//   3  the output from the coefficients' box means and the guide: q = mean a · I + mean b, clamped 0 to 1
// The mask's edge then runs where the guide's edges run, at the guide's resolution; where the guide is flat
// (its variance under ε) the mask is only averaged.

uniform sampler2D tex0;     // stage 0: the guide (RGB); 1: the field to average; 2: the means; 3: the coefficients' means
uniform sampler2D tex1;     // stage 0: the mask (R); 3: the guide (RGB)
uniform int stage;
uniform vec2 direction;     // stage 1: (1, 0) or (0, 1)
uniform int radius;         // stage 1: pixels each side
uniform float epsilon;      // stage 2: the guide's variance below which the mask is only averaged

in vec2 texCoord;
out vec4 fragColor;

float luminance(vec3 rgb) {
    return dot(rgb, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    if (stage == 0) {
        float I = luminance(texture(tex0, texCoord).rgb);
        float p = texture(tex1, texCoord).r;
        fragColor = vec4(I, p, I * I, I * p);
    } else if (stage == 1) {
        ivec2 size = textureSize(tex0, 0);
        ivec2 center = ivec2(texCoord * vec2(size));
        ivec2 step = ivec2(direction);
        vec4 sum = vec4(0.0);
        for (int k = -radius; k <= radius; k++) {
            sum += texelFetch(tex0, clamp(center + step * k, ivec2(0), size - 1), 0);
        }
        fragColor = sum / float(2 * radius + 1);
    } else if (stage == 2) {
        vec4 m = texture(tex0, texCoord);
        float variance = m.z - m.x * m.x;
        float covariance = m.w - m.x * m.y;
        float a = covariance / (variance + epsilon);
        fragColor = vec4(a, m.y - a * m.x, 0.0, 0.0);
    } else {
        vec2 ab = texture(tex0, texCoord).rg;
        float I = luminance(texture(tex1, texCoord).rgb);
        fragColor = vec4(clamp(ab.x * I + ab.y, 0.0, 1.0), 0.0, 0.0, 0.0);
    }
}
