#version 460 core

// Gradient Flow Shader
// Per pixel, the brightness change along the brightness gradient, averaged over three mip levels:
// the motion along the gradient, as a strength. Times `strength`, with `threshold` taken off its
// length. Ported from ofxFlowTools ftOpticalFlowShader

uniform sampler2D tex0;  // Previous luminance, mipmapped
uniform sampler2D tex1;  // Current luminance, mipmapped

uniform vec2 offset;     // Gradient sample offset (uv)
uniform float threshold; // Taken off the length, after strength
uniform vec2 strength;   // x and y multiplier (negative inverts)

const int numLevels = 3;   // Number of mip levels to sample (1-4)
const vec4 levelWeights = vec4(0.25f, 0.5f, 1.0f, 1.0f); // Weights for each level

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.0001

vec2 computeFlowAtLod(vec2 st, vec2 off, float lod) {
    vec2 off_x = vec2(off.x, 0.0);
    vec2 off_y = vec2(0.0, off.y);

    // Temporal difference at this LOD
    float scr_dif = textureLod(tex0, st, lod).x - textureLod(tex1, st, lod).x;

    // Spatial gradient at this LOD
    float gradx = textureLod(tex1, st + off_x, lod).x - textureLod(tex1, st - off_x, lod).x;
    gradx += textureLod(tex0, st + off_x, lod).x - textureLod(tex0, st - off_x, lod).x;

    float grady = textureLod(tex1, st + off_y, lod).x - textureLod(tex1, st - off_y, lod).x;
    grady += textureLod(tex0, st + off_y, lod).x - textureLod(tex0, st - off_y, lod).x;

    float gradmag = sqrt(gradx * gradx + grady * grady + TINY);

    vec2 flow;
    flow.x = scr_dif * (gradx / gradmag);
    flow.y = scr_dif * (grady / gradmag);

    return flow;
}

void main() {
    vec2 st = texCoord;
    vec2 totalFlow = vec2(0.0);
    float totalWeight = 0.0;

    // LOD 0 = full res, LOD 1 = half, LOD 2 = quarter
    for (int i = 0; i < numLevels && i < 4; i++) {
        float lod = float(i);
        float scale = pow(2.0, lod);  // Offset scales with LOD

        vec2 flow = computeFlowAtLod(st, offset * scale, lod);
        float weight = levelWeights[i];

        totalFlow += flow * weight;
        totalWeight += weight;
    }

    vec2 flow = totalFlow / (totalWeight + TINY) * strength;

    // The threshold off the length; the direction stays
    float magnitude = length(flow);
    float kept = max(magnitude - threshold, 0.0);
    flow = magnitude > TINY ? flow / magnitude * kept : vec2(0.0);

    fragColor = vec4(flow, 0.0, 1.0);
}
