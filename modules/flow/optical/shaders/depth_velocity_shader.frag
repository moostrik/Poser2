#version 460 core

// Depth Velocity Shader
// One frame of the depth track, in disparity (1 / depth): the track the frame before, carried from
// where the pixel was `displacement` (uv per frame) back, predicted on by its rate and corrected by
// the new measurement, an alpha–beta filter. The measurement is the mean disparity over a window of
// depth samples, of those within `tolerance` of the pixel's depth (not the background beside the
// player); it corrects by alpha and beta times the share of the window it holds, so a thin window
// corrects little. Further than `tolerance` from the prediction, or without a track there, the track
// starts over with no rate (a limb passing over the pixel). Without depth at the pixel or without a
// span, it is empty. z is the rate over the metres one view width spans there, `depth × view_span`,
// in view widths per frame. Depth is read nearest, so an edge never reads as a distance in between;
// the track is read linear between the texels that hold one of the same surface, within `tolerance`
// of the measurement, so a track is never carried across an edge. With `dead_zone` above 0, z is 0
// while the tracked depth stays within it of its anchor; past it, z at its full speed, and the anchor
// follows at `dead_zone` behind (a backlash); a track that starts over is anchored anew. The numpy
// reference is depth_velocity.track_depth

uniform sampler2D tex0;  // Depth now R (m)
uniform sampler2D tex1;  // The track the frame before: z, disparity (1/m), rate (1/m per second), anchor (m)
uniform sampler2D displacement;     // uv per frame (RG), with warp
uniform bool warp;
uniform vec2 depth_step; // uv between two depth samples
uniform int window;      // Radius in depth samples
uniform float tolerance; // Depth further than this is another surface (m)
uniform float alpha;     // Weight of a full window's measurement on the disparity
uniform float delta_time;           // Seconds since the last depth
uniform float view_span; // Metres one view width spans per metre of depth
uniform float dead_zone; // Depth the track wanders within its anchor without z (m); 0 is none

in vec2 texCoord;
out vec4 fragColor;

float depth_at(vec2 uv) {
    ivec2 size = textureSize(tex0, 0);
    ivec2 texel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
    return texelFetch(tex0, texel, 0).r;
}

// The track's disparity, rate and anchor at `position` (texels, centres at integers), of the texels
// that hold one of the measurement's surface, within `tolerance` of it; none of them: no track
vec3 carried(vec2 position, float measured_depth) {
    ivec2 size = textureSize(tex1, 0);
    ivec2 base = ivec2(floor(position));
    vec2 f = position - vec2(base);
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            vec4 texel = texelFetch(tex1, clamp(base + ivec2(dx, dy), ivec2(0), size - 1), 0);
            bool same = texel.g > 0.0 && abs(1.0 / texel.g - measured_depth) <= tolerance;
            float weight = (dx == 0 ? 1.0 - f.x : f.x) * (dy == 0 ? 1.0 - f.y : f.y) * float(same);
            sum += weight * texel.gba;
            total += weight;
        }
    }
    return total > 1e-9 ? sum / total : vec3(0.0);
}

void main() {
    float center = depth_at(texCoord);
    if (center <= 0.0 || view_span <= 0.0) {
        fragColor = vec4(0.0);
        return;
    }

    float total = 0.0;
    int kept = 0;
    for (int j = -window; j <= window; j++) {
        for (int i = -window; i <= window; i++) {
            float sample_depth = depth_at(texCoord + vec2(i, j) * depth_step);
            if (sample_depth > 0.0 && abs(sample_depth - center) <= tolerance) {
                total += 1.0 / sample_depth;
                kept++;
            }
        }
    }
    float side = float(2 * window + 1);
    float measurement = total / float(max(kept, 1));
    float share = float(kept) / (side * side);

    vec2 moved = warp ? texture(displacement, texCoord).xy : vec2(0.0);
    vec2 size = vec2(textureSize(tex1, 0));
    vec3 track = carried(texCoord * size - 0.5 - moved * size, 1.0 / measurement);
    float predicted = track.x > 0.0 ? track.x + track.y * delta_time : 0.0;

    if (predicted <= 0.0 || abs(1.0 / measurement - 1.0 / predicted) > tolerance) {
        fragColor = vec4(0.0, measurement, 0.0, 1.0 / measurement);
        return;
    }
    float a = alpha * share;
    float b = a * a / (2.0 - a);
    float residual = measurement - predicted;
    float disparity = predicted + a * residual;
    float rate = track.y + b * residual / delta_time;
    float z = rate * delta_time / (disparity * view_span);

    float tracked = 1.0 / disparity;
    float anchor = track.z > 0.0 ? track.z : tracked;
    if (dead_zone > 0.0) {
        float offset = tracked - anchor;
        if (abs(offset) > dead_zone) {
            anchor = tracked - sign(offset) * dead_zone;
        } else {
            z = 0.0;
        }
    } else {
        anchor = tracked;
    }
    fragColor = vec4(z, disparity, rate, anchor);
}
