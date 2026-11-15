#version 460 core

// Depth in metres as a picture. The range is depthai's `colorizeDepthFrame`: a log scale from nearDepth to farDepth,
// through the colormap `map` (jet is depthai's). No depth (0) is transparent.

#include "colormap.glsl"

uniform sampler2D depth;
uniform sampler2D image;
uniform float nearDepth;
uniform float farDepth;
uniform int mode;           // DepthDrawMode
uniform int map;            // gl.Colormap

const int GREY = 0;
const int COLOR = 1;
const int OVER_IMAGE = 2;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float meters = texture(depth, texCoord).r;
    bool valid = meters > 0.0;
    float t = clamp(log(max(meters, nearDepth) / nearDepth) / log(farDepth / nearDepth), 0.0, 1.0);

    if (mode == OVER_IMAGE) {
        vec3 under = texture(image, texCoord).rgb;
        fragColor = vec4(valid ? mix(under, colormap(map, t), 0.5) : under, 1.0);
    } else if (!valid) {
        fragColor = vec4(0.0);
    } else if (mode == GREY) {
        fragColor = vec4(vec3(1.0 - t), 1.0);
    } else {
        fragColor = vec4(colormap(map, t), 1.0);
    }
}
