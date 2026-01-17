#version 460 core

// Trail Shader - general temporal smoothing
// Ported from ofxFlowTools ftBridgeShader.h
// Blends previous trail with new input for temporal smoothing

uniform sampler2D tex0;     // Previous trail
uniform sampler2D tex1;     // New input
uniform float trailWeight;  // Trail weight (0=replace, 0.99=keep trail)
uniform float newWeight;    // New input weight/scale (default 1.0)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 prev = texture(tex0, texCoord);
    vec4 new_val = texture(tex1, texCoord) * newWeight;

    // Weighted blend: trail * trailWeight + new * newWeight
    vec4 result = (prev * trailWeight) + new_val;

    fragColor = result;
}
