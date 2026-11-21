#version 460 core

layout(binding = 0) uniform sampler2D src;
uniform float threshold;
uniform float detailBoost;
uniform float radius;
uniform bool invert;
#define texelSize (1.0 / vec2(textureSize(src, 0)))

#include "color.glsl"

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Sample center pixel luminance
    vec3 centerColor = texture(src, texCoord).rgb;
    float centerLum = luminance(centerColor);

    // Compute local mean and variance in a neighborhood
    float sum = 0.0;
    float sumSq = 0.0;
    float count = 0.0;

    // Sample in a cross pattern for efficiency (9 samples)
    for (float y = -1.0; y <= 1.0; y += 1.0) {
        for (float x = -1.0; x <= 1.0; x += 1.0) {
            vec2 offset = vec2(x, y) * texelSize * radius;
            vec3 sampleColor = texture(src, texCoord + offset).rgb;
            float sampleLum = luminance(sampleColor);
            sum += sampleLum;
            sumSq += sampleLum * sampleLum;
            count += 1.0;
        }
    }

    float localMean = sum / count;
    float localVariance = (sumSq / count) - (localMean * localMean);
    float localStdDev = sqrt(max(localVariance, 0.0));

    // Detail metric: high local contrast = detailed area (face)
    // Scale stdDev to 0-1 range (typical values are 0-0.3)
    float detail = clamp(localStdDev * 3.0, 0.0, 1.0);

    // Adaptive threshold: in detailed areas, use finer threshold
    // In uniform areas, use coarser threshold (more black/white)
    float adaptiveThreshold = mix(threshold, localMean, detail * detailBoost);

    // Apply threshold with slight smoothing
    float result = smoothstep(adaptiveThreshold - 0.05, adaptiveThreshold + 0.05, centerLum);

    // In detailed areas, preserve more mid-tones
    // Blend between hard threshold and original luminance based on detail
    float detailPreserve = mix(result, centerLum, detail * detailBoost * 0.5);

    // Final threshold to ensure black/white output
    float finalResult = step(0.5, detailPreserve);

    // Optional: soften edges slightly for anti-aliasing
    finalResult = smoothstep(0.4, 0.6, detailPreserve);

    if (invert) {
        finalResult = 1.0 - finalResult;
    }

    fragColor = vec4(vec3(finalResult), 1.0);
}
