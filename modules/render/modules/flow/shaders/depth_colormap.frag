#version 460 core

// Depth in meters as a picture. The range is depthai's `colorizeDepthFrame`: a log scale from nearDepth to farDepth,
// through the colormap `map` (jet is depthai's). No depth (0) is transparent. The depth's confidence is its green:
// COLOR_BY_CONFIDENCE draws the colormap as bright as it, CONFIDENCE draws it alone as grey from 0 black to 1 white.

#include "colormap.glsl"

layout(binding = 0) uniform sampler2D depth;
layout(binding = 1) uniform sampler2D image;
uniform float nearDepth;
uniform float farDepth;
uniform int mode;           // DepthDrawMode
uniform int map;            // gl.Colormap

const int GREY = 0;
const int COLOR = 1;
const int OVER_IMAGE = 2;
const int CONFIDENCE = 3;
const int COLOR_BY_CONFIDENCE = 4;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 sampled = texture(depth, texCoord).rg;
    float meters = sampled.r;
    float confidence = clamp(sampled.g, 0.0, 1.0);
    bool valid = meters > 0.0;
    float t = clamp(log(max(meters, nearDepth) / nearDepth) / log(farDepth / nearDepth), 0.0, 1.0);

    if (mode == OVER_IMAGE) {
        vec3 under = texture(image, texCoord).rgb;
        fragColor = vec4(valid ? mix(under, colormap(map, t), 0.5) : under, 1.0);
    } else if (!valid) {
        fragColor = vec4(0.0);
    } else if (mode == CONFIDENCE) {
        fragColor = vec4(vec3(confidence), 1.0);
    } else if (mode == COLOR_BY_CONFIDENCE) {
        fragColor = vec4(colormap(map, t) * confidence, 1.0);
    } else if (mode == GREY) {
        fragColor = vec4(vec3(1.0 - t), 1.0);
    } else {
        fragColor = vec4(colormap(map, t), 1.0);
    }
}
