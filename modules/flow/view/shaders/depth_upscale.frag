#version 460 core

// The depth read into the view, meters in red and its confidence in green, where it is the player's and the mask is:
// the four depth pixels around a view pixel interpolated bilinearly where they lie within `tolerance` (m) of each
// other, else the one it lies in, so an edge never reads as a depth in between; the confidence with the depth's own
// weights, or the one pixel's. Kept where the depth is from `near` to `far` and the mask is above 0, else 0 and 0. A
// 0, no depth, is never blended, and beyond the depth's grid is no depth.

#include "position.glsl"

uniform sampler2D depth;        // meters on the camera image in red, 0 for none; the confidence 0 to 1 in green
uniform sampler2D mask;         // the player's mask in the view (red, 0 to 1)
uniform vec4 roi;               // the view's window on the depth, x, y, width, height in 0..1, origin top-left
uniform float tolerance;
uniform float near;             // the player's depth range (m)
uniform float far;

in vec2 texCoord;
out vec4 fragColor;

vec2 fetch(ivec2 p, ivec2 size) {
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return vec2(0.0);
    }
    return texelFetch(depth, p, 0).rg;
}

void main() {
    ivec2 size = textureSize(depth, 0);
    vec2 uv = topLeft(roi.xy + topLeft(texCoord) * roi.zw);
    vec2 p = uv * vec2(size) - 0.5;
    vec2 corner = floor(p);
    vec2 f = p - corner;
    ivec2 i = ivec2(corner);
    vec2 d00 = fetch(i, size);
    vec2 d10 = fetch(i + ivec2(1, 0), size);
    vec2 d01 = fetch(i + ivec2(0, 1), size);
    vec2 d11 = fetch(i + ivec2(1, 1), size);
    float low = min(min(d00.r, d10.r), min(d01.r, d11.r));
    float high = max(max(d00.r, d10.r), max(d01.r, d11.r));
    vec2 read;
    if (low > 0.0 && high - low <= tolerance) {
        read = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
    } else {
        read = fetch(ivec2(floor(uv * vec2(size))), size);
    }
    float meters = read.r;
    bool kept = meters > 0.0 && meters >= near && meters <= far && texture(mask, texCoord).r > 0.0;
    fragColor = kept ? vec4(meters, read.g, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
}
