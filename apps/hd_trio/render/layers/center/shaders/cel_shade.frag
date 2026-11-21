#version 460 core

layout(binding = 0) uniform sampler2D src;
// Color correction
uniform float exposure;
uniform float gamma;
uniform float offset;
uniform float contrast;
// Cel shading
uniform int levels;
uniform float smoothness;
uniform float saturation;

in vec2 texCoord;
out vec4 fragColor;

#include "color.glsl"

void main() {
    vec4 color = texture(src, texCoord);
    vec2 texelSize = 1.0 / vec2(textureSize(src, 0));

    // Check if we're in a dark area - if so, blur first (only with the bands, levels > 0)
    float lum = luminance(color.rgb);
    float blurAmount = levels > 0 ? smoothstep(0.3, 0.05, lum) : 0.0;  // More blur in darker areas

    if (blurAmount > 0.01) {
        // 3x3 gaussian-ish blur for dark areas
        vec3 blurred = vec3(0.0);
        float weights[9] = float[](1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0);
        float totalWeight = 16.0;
        int idx = 0;
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                vec2 offset_uv = texCoord + vec2(x, y) * texelSize * 2.0;
                blurred += texture(src, offset_uv).rgb * weights[idx];
                idx++;
            }
        }
        blurred /= totalWeight;
        color.rgb = mix(color.rgb, blurred, blurAmount);
    }

    // === COLOR CORRECTION ===
    // Exposure: multiply then add offset
    vec3 corrected = color.rgb * exposure + offset;
    // Gamma correction
    corrected = pow(max(corrected, 0.0), vec3(1.0 / gamma));
    // Contrast: expand/compress around 0.5
    corrected = (corrected - 0.5) * contrast + 0.5;
    corrected = clamp(corrected, 0.0, 1.0);

    // === CEL SHADING ===
    vec3 hsv = rgb2hsv(corrected);
    float originalV = hsv.z;

    // Apply saturation
    hsv.y = clamp(hsv.y * saturation, 0.0, 1.0);

    // levels 0: no bands, the corrected color as it is
    if (levels <= 0) {
        fragColor = vec4(hsv2rgb(hsv), color.a);
        return;
    }

    // Quantize value into discrete levels with smooth transitions
    // Use perceptual (gamma) space for more even banding in darks
    float v = sqrt(hsv.z);  // Linear to perceptual
    float levelSize = 1.0 / float(levels);
    float bandPosition = fract(v / levelSize + 0.5);
    float smooth_t = smoothstep(0.5 - smoothness, 0.5 + smoothness, bandPosition);
    float lowerBand = floor(v / levelSize) * levelSize;
    float upperBand = lowerBand + levelSize;
    v = mix(lowerBand, upperBand, smooth_t);
    float posterizedV = v * v;  // Perceptual back to linear

    // Blend to original in dark areas to avoid blocky shadows
    float shadowBlend = smoothstep(0.0, 0.25, originalV);
    hsv.z = mix(originalV, posterizedV, shadowBlend);

    // Quantize hue into 12 bands (only for non-shadow areas)
    float hueSteps = 12.0;
    float quantizedHue = floor(hsv.x * hueSteps + 0.5) / hueSteps;
    hsv.x = mix(hsv.x, quantizedHue, shadowBlend);

    vec3 result = hsv2rgb(hsv);
    fragColor = vec4(result, color.a);
}