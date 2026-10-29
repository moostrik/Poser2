#version 460 core

uniform sampler2D tex;
uniform vec3 targetColor;
uniform float strength;

in vec2 texCoord;
out vec4 fragColor;

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec4 color = texture(tex, texCoord);
    vec3 srcHsv = rgb2hsv(color.rgb);
    vec3 targetHsv = rgb2hsv(targetColor);

    // Shift hue toward target, boost saturation toward target
    float newHue = mix(srcHsv.x, targetHsv.x, strength);
    float newSat = mix(srcHsv.y, max(srcHsv.y, targetHsv.y * 0.8), strength);

    // Keep original value (luminance)
    vec3 result = hsv2rgb(vec3(newHue, newSat, srcHsv.z));

    fragColor = vec4(result, color.a);
}
