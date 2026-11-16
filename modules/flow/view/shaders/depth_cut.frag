#version 460 core

// The player cut on the depth's own pixels, 1 in red where the depth pixel a view pixel lies in is from `near` to
// `far` (m), 0 elsewhere: no depth (0) and beyond the depth's grid are not the player. No blend, so the cut keeps the
// depth's own pixels, the outline the guided filter then moves onto the image's.

#include "position.glsl"

uniform sampler2D depth;        // metres on the camera image, 0 for none
uniform vec4 roi;               // the view's window on the depth, x, y, width, height in 0..1, origin top-left
uniform float near;
uniform float far;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 size = textureSize(depth, 0);
    vec2 uv = topLeft(roi.xy + topLeft(texCoord) * roi.zw);
    ivec2 p = ivec2(floor(uv * vec2(size)));
    float metres = any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size)) ? 0.0 : texelFetch(depth, p, 0).r;
    float player = metres > 0.0 && metres >= near && metres <= far ? 1.0 : 0.0;
    fragColor = vec4(player, 0.0, 0.0, 1.0);
}
