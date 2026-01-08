#version 460 core

uniform sampler2D tex0;       // Flow texture (RG = x,y displacement)
uniform float scale;          // Scale factor for flow magnitude
uniform float gamma;          // Gamma correction for visualization
uniform float noiseThreshold; // Threshold to filter out noise

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 flow = texture(tex0, texCoord).rg;

    float rawMagnitude = length(flow);

    // Smooth falloff instead of hard threshold
    // Creates gradual fade from 0 to 1 over range [threshold*0.5, threshold*2]
    float noiseFactor = smoothstep(noiseThreshold * 0.5, noiseThreshold * 2.0, rawMagnitude);

    if (noiseFactor < 0.01) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Scale flow for visualization
    vec3 scaledFlow = vec3(flow * scale, 0.0);

    // Calculate magnitude with gamma correction and clamp
    float magnitude = clamp(pow(rawMagnitude * scale, gamma), 0.0, 1.0);

    // Encode: RGB = flow direction (centered at 0.5), clamped to [0,1]
    vec3 color = clamp(scaledFlow + vec3(0.5), 0.0, 1.0);

    // Apply noise factor to alpha for smooth fade
    fragColor = vec4(color, magnitude * noiseFactor);
}