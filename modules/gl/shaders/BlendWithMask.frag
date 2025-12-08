#version 460 core

uniform sampler2D prev;      // Previous blended mask
uniform sampler2D curr;      // Current mask (low-res)
uniform float blend;         // Temporal blend factor
uniform vec2 texelSize;      // 1.0 / texture_dimensions for blur
uniform float blurRadius;    // Blur radius multiplier

in vec2 texCoord;
out vec4 fragColor;

// Gaussian blur weights for 9-tap kernel
const float gaussianWeights[9] = float[](
    0.081149, 0.076883, 0.070266, 0.061948, 0.041312,
    0.061948, 0.070266, 0.076883, 0.081149
);

float dilateAndBlurRed(sampler2D tex, vec2 uv) {
    float r = blurRadius;
    float maxVal = 0.0;

    // Sample in a circle pattern and take maximum (dilation)
    maxVal = max(maxVal, texture(tex, uv).r);
    maxVal = max(maxVal, texture(tex, uv + vec2(-2.0 * texelSize.x * r, 0.0)).r);
    maxVal = max(maxVal, texture(tex, uv + vec2( 2.0 * texelSize.x * r, 0.0)).r);
    maxVal = max(maxVal, texture(tex, uv + vec2(0.0, -2.0 * texelSize.y * r)).r);
    maxVal = max(maxVal, texture(tex, uv + vec2(0.0,  2.0 * texelSize.y * r)).r);
    maxVal = max(maxVal, texture(tex, uv + vec2(-1.0 * texelSize.x * r, 0.0)).r);
    maxVal = max(maxVal, texture(tex, uv + vec2( 1.0 * texelSize.x * r, 0.0)).r);
    maxVal = max(maxVal, texture(tex, uv + vec2(0.0, -1.0 * texelSize.y * r)).r);
    maxVal = max(maxVal, texture(tex, uv + vec2(0.0,  1.0 * texelSize.y * r)).r);

    // Apply blur to smooth the expanded mask
    float result = 0.0;
    result += 0.25 * maxVal;
    result += 0.15 * texture(tex, uv + vec2(-1.0 * texelSize.x * r, 0.0)).r;
    result += 0.15 * texture(tex, uv + vec2( 1.0 * texelSize.x * r, 0.0)).r;
    result += 0.15 * texture(tex, uv + vec2(0.0, -1.0 * texelSize.y * r)).r;
    result += 0.15 * texture(tex, uv + vec2(0.0,  1.0 * texelSize.y * r)).r;
    result += 0.075 * texture(tex, uv + vec2(-2.0 * texelSize.x * r, 0.0)).r;
    result += 0.075 * texture(tex, uv + vec2( 2.0 * texelSize.x * r, 0.0)).r;
    result += 0.075 * texture(tex, uv + vec2(0.0, -2.0 * texelSize.y * r)).r;
    result += 0.075 * texture(tex, uv + vec2(0.0,  2.0 * texelSize.y * r)).r;

    return result;
}

void main() {
    // Sample previous blended mask (already high-res) - only red channel
    float prevMask = texture(prev, texCoord).r;

    // Dilate and blur current mask - expands edges and smooths
    float blurredCurr = dilateAndBlurRed(curr, texCoord);

    // Apply power curve and threshold to current frame for sharp edges
    // blurredCurr = pow(blurredCurr, 1.5);
    blurredCurr = smoothstep(0.2, 0.5, blurredCurr);

    // Temporal blend: mix sharp current with previous (fade out over time)
    float blendedMask = mix(prevMask, blurredCurr, blend);
    blendedMask = clamp(blendedMask, 0.0, 1.0);

    // Output to red channel (mask format)
    fragColor = vec4(blendedMask, blendedMask, blendedMask, 1.0);
}