#version 460 core

// The camera's depth inside the view's box on its own pixels, metres in red: the depth pixel a view pixel lies in
// where it is from `near` to `far` (m), 0 elsewhere; no depth (0) and beyond the depth's grid are 0. No blend, so it
// keeps the depth's own pixels, the outline the guided filter then moves onto the image's.

#include "position.glsl"

uniform sampler2D depth;        // metres on the camera image, 0 for none
uniform vec4 roi;               // the box's sides on the depth, x, y, width, height in 0..1, origin top-left
uniform float near;             // the box's front (m)
uniform float far;              // its back (m)

in vec2 texCoord;
out vec4 fragColor;

void main() {
    ivec2 size = textureSize(depth, 0);
    vec2 uv = topLeft(roi.xy + topLeft(texCoord) * roi.zw);
    ivec2 p = ivec2(floor(uv * vec2(size)));
    float metres = any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size)) ? 0.0 : texelFetch(depth, p, 0).r;
    float inside = metres > 0.0 && metres >= near && metres <= far ? metres : 0.0;
    fragColor = vec4(inside, 0.0, 0.0, 1.0);
}
