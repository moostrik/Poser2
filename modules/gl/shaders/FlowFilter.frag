#version 460 core

uniform sampler2D tex0;      // Flow texture (RG = x,y displacement)
uniform float scale;          // Scale factor for flow magnitude
uniform float gamma;          // Gamma correction for visualization
uniform float noiseThreshold; // Threshold to filter out noise

in vec2 texCoord;
out vec4 fragColor;

const float PI = 3.14159265359;

// Convert flow vector to HSV color (direction = hue, magnitude = brightness)
vec3 flowToColor(vec2 flow) {
    float rawMagnitude = length(flow);

    // Filter noise: use RAW magnitude before scaling
    if (rawMagnitude < noiseThreshold) {
        return vec3(0.0);
    }

    // NOW scale for visualization
    float magnitude = rawMagnitude * scale;

    float angle = atan(flow.y, flow.x);

    // Normalize angle to [0, 1]
    float hue = (angle + PI) / (2.0 * PI);

    // Apply gamma to magnitude for brightness control
    float value = pow(clamp(magnitude, 0.0, 1.0), gamma);

    // HSV to RGB conversion
    float h = hue * 6.0;
    float c = value;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));

    vec3 rgb;
    if (h < 1.0)      rgb = vec3(c, x, 0.0);
    else if (h < 2.0) rgb = vec3(x, c, 0.0);
    else if (h < 3.0) rgb = vec3(0.0, c, x);
    else if (h < 4.0) rgb = vec3(0.0, x, c);
    else if (h < 5.0) rgb = vec3(x, 0.0, c);
    else              rgb = vec3(c, 0.0, x);

    return rgb;
}

// Bilateral filter for smoothing while preserving edges
vec3 bilateralFilter(sampler2D tex, vec2 uv, float spatialSigma, float rangeSigma) {
    vec2 texelSize = 1.0 / textureSize(tex, 0);
    vec2 centerFlow = texture(tex, uv).rg;
    float centerMag = length(centerFlow); // RAW magnitude

    // Early exit for noise using RAW magnitude
    if (centerMag < noiseThreshold) {
        return vec3(0.0);
    }

    vec3 sum = vec3(0.0);
    float weightSum = 0.0;

    int radius = 2;
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            vec2 sampleFlow = texture(tex, uv + offset).rg;

            // Spatial weight
            float spatialDist = length(vec2(x, y));
            float spatialWeight = exp(-spatialDist * spatialDist / (2.0 * spatialSigma * spatialSigma));

            // Range weight (flow similarity)
            float flowDiff = length(sampleFlow - centerFlow);
            float rangeWeight = exp(-flowDiff * flowDiff / (2.0 * rangeSigma * rangeSigma));

            float weight = spatialWeight * rangeWeight;
            sum += flowToColor(sampleFlow) * weight;
            weightSum += weight;
        }
    }

    return weightSum > 0.0 ? sum / weightSum : vec3(0.0);
}

void main() {
    vec2 flow = texture(tex0, texCoord).rg;

    // Option 1: Direct color mapping (fast)
    vec3 color = flowToColor(flow);

    // Option 2: With bilateral filtering (smoother, slower)
    // vec3 color = bilateralFilter(tex0, texCoord, 2.0, 0.1);

    fragColor = vec4(color, 1.0);
}