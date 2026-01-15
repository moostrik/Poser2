#version 460 core

// HSV Color Adjustment Shader
// Ported from ofxFlowTools ftHSVShader.h
// Adjusts hue, saturation, and value of RGB colors

uniform sampler2D tex0;
uniform float hue;         // Hue shift (-0.5 to 0.5)
uniform float saturation;  // Saturation multiplier (0.0 to 5.0)
uniform float value;       // Value/brightness multiplier (0.0 to 2.0)

in vec2 texCoord;
out vec4 fragColor;

// RGB to HSV conversion
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec4 color = texture(tex0, texCoord);

    // Convert to HSV
    vec3 hsv = rgb2hsv(color.rgb);

    // Adjust HSV channels
    hsv.x = fract(hsv.x + hue);           // Hue shift (wrap around)
    hsv.y = clamp(hsv.y * saturation, 0.0, 1.0);  // Saturation multiply
    hsv.z = clamp(hsv.z * value, 0.0, 1.0);       // Value multiply

    // Convert back to RGB
    vec3 rgb = hsv2rgb(hsv);

    fragColor = vec4(rgb, color.a);  // Preserve alpha
}
