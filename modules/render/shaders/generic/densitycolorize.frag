#version 460 core

precision highp float;

// Fullscreen quad texture coordinates
in vec2 texCoord;

// Output
out vec4 fragColor;

// Input textures
uniform sampler2D uDensity;     // Density field (RGBA16F) - each channel is a different track

// Track colors - up to 4 tracks mapped to RGBA channels
uniform vec4 uColors[4];

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
    vec4 den = texture(uDensity, texCoord);

    // Map each density channel to its corresponding track color
    // R channel -> track 0, G -> track 1, B -> track 2, A -> track 3
    vec3 result = vec3(0.0);
    result += den.r * uColors[0].rgb;
    result += den.g * uColors[1].rgb;
    result += den.b * uColors[2].rgb;
    result += den.a * uColors[3].rgb;

    // Normalize brightness using HSV (clamp V to 1.0 if overbright)
    vec3 hsv = rgb2hsv(result);
    if (hsv.z > 1.0) {
        hsv.z = 1.0;
        result = hsv2rgb(hsv);
    }

    // Alpha from max of densities (for transparency blending)
    float alpha = max(max(den.r, den.g), max(den.b, den.a));

    fragColor = vec4(result, alpha);
}
