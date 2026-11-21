#version 460 core

layout(binding = 0) uniform sampler2D src;
uniform int radius;         // pixels each side
uniform float sigmaSpace;   // pixels
uniform float sigmaColor;   // color difference, 0..1

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 last = textureSize(src, 0) - 1;
    ivec2 center = ivec2(texCoord * vec2(last + 1));
    vec4 c0 = texelFetch(src, center, 0);
    float space = -0.5 / (sigmaSpace * sigmaSpace);
    float range = -0.5 / (sigmaColor * sigmaColor);
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            vec4 c = texelFetch(src, clamp(center + ivec2(x, y), ivec2(0), last), 0);
            vec3 d = c.rgb - c0.rgb;
            float w = exp(float(x * x + y * y) * space + dot(d, d) * range);
            sum += c * w;
            total += w;
        }
    }
    fragColor = sum / total;
}
