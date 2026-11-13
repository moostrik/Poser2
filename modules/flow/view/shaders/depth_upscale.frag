#version 460 core

// The depth read into the view, one of two ways (`DepthRead`), metres in red; a 0, no depth, is never blended,
// and beyond the depth's grid is no depth.
//   SURFACE    the four depth pixels around a view pixel interpolated bilinearly where they lie within `tolerance`
//              (m) of each other, else the one it lies in, so an edge never reads as a depth in between.
//   MASK       as SURFACE where that depth is on the side the mask gives the pixel: the player's (a depth from
//              `near` to `far`) where the mask is, anything else (none included) where it is not; else the nearest
//              depth pixel on that side within `reach`, else SURFACE's.

const int SURFACE = 0;
const int MASK = 1;
const int MAX_REACH = 9;        // MASK: the most depth pixels it looks, each way (`guided_radius` at most 8, + 0.5)

uniform sampler2D depth;        // metres on the camera image, 0 for none
uniform sampler2D mask;         // MASK: the player's mask in the view (red, 0 to 1)
uniform vec4 roi;               // the view's window on the depth, x, y, width, height in texture coordinates
uniform float tolerance;
uniform int mode;
uniform float near;             // MASK: the player's depth range (m)
uniform float far;
uniform float reach;            // MASK: how far it looks for a depth on the mask's side (depth pixels)

in vec2 texCoord;
out vec4 fragColor;

float fetch(ivec2 p, ivec2 size) {
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, size))) {
        return 0.0;
    }
    return texelFetch(depth, p, 0).r;
}

bool player(float d) {
    return d > 0.0 && d >= near && d <= far;
}

// MASK: the read where it is on the mask's side, else the nearest depth pixel within `reach` that is, else the read
float mask_side(float metres, vec2 p, ivec2 size) {
    bool inside = texture(mask, texCoord).r > 0.5;
    if (player(metres) == inside) {
        return metres;
    }
    int r = min(int(ceil(reach + 0.5)), MAX_REACH);
    ivec2 center = ivec2(floor(p + 0.5));
    float best = reach * reach;
    float found = metres;
    for (int y = -MAX_REACH; y <= MAX_REACH; y++) {
        if (abs(y) > r) continue;
        for (int x = -MAX_REACH; x <= MAX_REACH; x++) {
            if (abs(x) > r) continue;
            ivec2 at = center + ivec2(x, y);
            float d = fetch(at, size);
            if (player(d) != inside) continue;
            vec2 offset = vec2(at) - p;
            float distance = dot(offset, offset);
            if (distance <= best) {
                best = distance;
                found = d;
            }
        }
    }
    return found;
}

void main() {
    ivec2 size = textureSize(depth, 0);
    vec2 uv = roi.xy + texCoord * roi.zw;
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
    if (mode == MASK) {
        metres = mask_side(metres, p, size);
    }
    fragColor = vec4(metres, 0.0, 0.0, 1.0);
}
