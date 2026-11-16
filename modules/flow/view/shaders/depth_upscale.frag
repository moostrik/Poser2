#version 460 core

// The depth read into the view, metres in red, where it is the player's and the mask is: the four depth pixels
// around a view pixel interpolated bilinearly where they lie within `tolerance` (m) of each other, else the one it
// lies in, so an edge never reads as a depth in between; kept where it is from `near` to `far` and the mask is
// above 0, else 0. A 0, no depth, is never blended, and beyond the depth's grid is no depth.

#include "position.glsl"

uniform sampler2D depth;        // metres on the camera image, 0 for none
uniform sampler2D mask;         // the player's mask in the view (red, 0 to 1)
uniform vec4 roi;               // the view's window on the depth, x, y, width, height in 0..1, origin top-left
uniform float tolerance;
uniform float near;             // the player's depth range (m)
uniform float far;

in vec2 texCoord;
out vec4 fragColor;

float fetch(ivec2 p, ivec2 size) {
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return 0.0;
    }
    return texelFetch(depth, p, 0).r;
}

void main() {
    ivec2 size = textureSize(depth, 0);
    vec2 uv = topLeft(roi.xy + topLeft(texCoord) * roi.zw);
    vec2 p = uv * vec2(size) - 0.5;
    vec2 corner = floor(p);
    vec2 f = p - corner;
    ivec2 i = ivec2(corner);
    float d00 = fetch(i, size);
    float d10 = fetch(i + ivec2(1, 0), size);
    float d01 = fetch(i + ivec2(0, 1), size);
    float d11 = fetch(i + ivec2(1, 1), size);
    float low = min(min(d00, d10), min(d01, d11));
    float high = max(max(d00, d10), max(d01, d11));
    float metres;
    if (low > 0.0 && high - low <= tolerance) {
        metres = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
    } else {
        metres = fetch(ivec2(floor(uv * vec2(size))), size);
    }
    bool kept = metres > 0.0 && metres >= near && metres <= far && texture(mask, texCoord).r > 0.0;
    fragColor = vec4(kept ? metres : 0.0, 0.0, 0.0, 1.0);
}
