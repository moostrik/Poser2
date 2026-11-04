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
// follows at `dead_zone` behind (a backlash); a track that starts over is anchored anew. With
// `restart_frames` above 1, it also starts over where the depth at the pixel, in place, is further
// than `tolerance` from its depth 2 to `restart_frames` frames before (`history`): a change that
// ramps past `tolerance` over a few frames, as the background a moving limb uncovers.
// With `slope_half` above 0, z fades where the depth surface is steep: its slope over the window
// (depth over the metres across, from the samples `window` samples to either side, or the center
// where one has no depth) weighs z by 1 / (1 + (slope / slope_half)²), half at `slope_half`. An edge
// stereo smears over a few samples is steep, a palm facing the camera is flat.
// Catching up, with `catch_up_smoothing` above 0: Emas of the miss of the prediction, signed and its size;
// where the miss keeps one way (the signed mean as large as the size, over twice the dead zone) and the
// image moves (the 3×3 displacement past `motion_floor`), the time the track follows over blends from
// `smoothing` toward `catch_up_smoothing`; and the carry takes each texel's depth a frame on by its rate. Run
// twice on the same inputs: with `catch_up_pass` it writes this frame's Emas, without it the track.
// The numpy reference is depth_velocity.track_depth

uniform sampler2D tex0;  // Depth now R (m)
uniform sampler2D tex1;  // The track the frame before: z, disparity (1/m), rate (1/m per second), anchor (m)
uniform sampler2D tex2;  // The catch-up the frame before: Emas of the miss, signed and its size (m)
uniform sampler2D displacement;     // uv per frame (RG), with warp
uniform sampler2D history;          // The depth 1 to 4 frames before, in place (m), at the track's size
uniform int restart_frames;         // Frames back a depth change past `tolerance` starts the track over; 1 is the frame before only
uniform bool warp;
uniform bool catch_up_pass;         // Write the catch-up instead of the track
uniform vec2 depth_step; // uv between two depth samples
uniform int window;      // Radius in depth samples
uniform float tolerance; // Depth further than this is another surface (m)
uniform float alpha;     // Weight of a full window's measurement on the disparity
uniform float smoothing;   // Time the track follows over (ms), `smooth` in Python (`smooth` is a GLSL keyword); 0 takes the measurement whole
uniform float catch_up_smoothing;       // Time it follows over while catching up (ms); 0 is never
uniform float ema;       // Weight of a new miss on its Emas
uniform float motion_floor;         // Displacement the image must move by to catch up (px per frame)
uniform float delta_time;           // Seconds since the last depth
uniform float view_span; // Metres one view width spans per metre of depth
uniform float dead_zone; // Depth the track wanders within its anchor without z (m); 0 is none
uniform float slope_half;           // Depth slope (m depth per m across) at which z is halved; 0 is no fade

in vec2 texCoord;
out vec4 fragColor;

float depth_at(vec2 uv) {
    ivec2 size = textureSize(tex0, 0);
    ivec2 texel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
    return texelFetch(tex0, texel, 0).r;
}

// The depth change per sample along `offset` (uv, `samples` depth samples): between the samples on
// either side, or the center and the one that has depth; 0 without either
float depth_change(vec2 offset, float samples, float center) {
    float ahead = depth_at(texCoord + offset);
    float behind = depth_at(texCoord - offset);
    if (ahead > 0.0 && behind > 0.0) return (ahead - behind) / (2.0 * samples);
    if (ahead > 0.0) return (ahead - center) / samples;
    if (behind > 0.0) return (center - behind) / samples;
    return 0.0;
}

// The weight of z on the depth surface's slope: 1 where it faces the camera, half at `slope_half`
float slope_weight(float center) {
    if (slope_half <= 0.0) return 1.0;
    vec2 size = vec2(textureSize(tex1, 0));
    float samples = float(max(window, 1));
    float across = center * view_span;     // metres one uv across spans here
    float slope_x = depth_change(vec2(depth_step.x * samples, 0.0), samples, center) / (across * depth_step.x);
    float slope_y = depth_change(vec2(0.0, depth_step.y * samples), samples, center) / (across * depth_step.y * size.y / size.x);
    float ratio = (slope_x * slope_x + slope_y * slope_y) / (slope_half * slope_half);
    return 1.0 / (1.0 + ratio);
}

// The track's disparity, rate and anchor (xyz) at `position` (texels, centres at integers), of the
// texels that hold one of the measurement's surface, within `tolerance` of it, each taken `ahead`
// seconds on by its rate; their catch-up in `catch_up`. None of them: no track
vec3 carried(vec2 position, float measured_depth, float ahead, out vec2 catch_up) {
    ivec2 size = textureSize(tex1, 0);
    ivec2 base = ivec2(floor(position));
    vec2 f = position - vec2(base);
    vec3 sum = vec3(0.0);
    vec2 sum_catch_up = vec2(0.0);
    float total = 0.0;
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            ivec2 at = clamp(base + ivec2(dx, dy), ivec2(0), size - 1);
            vec4 texel = texelFetch(tex1, at, 0);
            bool same = texel.g > 0.0 && abs(1.0 / (texel.g + texel.b * ahead) - measured_depth) <= tolerance;
            float weight = (dx == 0 ? 1.0 - f.x : f.x) * (dy == 0 ? 1.0 - f.y : f.y) * float(same);
            sum += weight * texel.gba;
            sum_catch_up += weight * texelFetch(tex2, at, 0).rg;
            total += weight;
        }
    }
    catch_up = total > 1e-9 ? sum_catch_up / total : vec2(0.0);
    return total > 1e-9 ? sum / total : vec3(0.0);
}

// How far the track is catching up, 0 to 1: the signed Ema over the size's and twice the dead zone,
// from 0.3 to 1 at 0.7, smoothly
float catch_up_share(float signed_mean, float size_mean) {
    float index = abs(signed_mean) / (size_mean + 2.0 * max(dead_zone, 0.005));
    return smoothstep(0.3, 0.7, index);
}

// Whether the image moves here: the largest displacement of the 3×3 texels past `motion_floor` (px)
bool moving(vec2 size) {
    ivec2 at = ivec2(floor(texCoord * size));
    ivec2 last_texel = ivec2(size) - 1;
    float largest = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 d = texelFetch(displacement, clamp(at + ivec2(i, j), ivec2(0), last_texel), 0).xy * size;
            largest = max(largest, length(d));
        }
    }
    return largest > motion_floor;
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

    bool catching = catch_up_smoothing > 0.0;
    vec2 moved = warp ? texture(displacement, texCoord).xy : vec2(0.0);
    vec2 size = vec2(textureSize(tex1, 0));
    vec2 last;
    vec3 track = carried(texCoord * size - 0.5 - moved * size, 1.0 / measurement, catching ? delta_time : 0.0, last);
    float predicted = track.x > 0.0 ? track.x + track.y * delta_time : 0.0;
    float miss = predicted > 0.0 ? 1.0 / measurement - 1.0 / predicted : 0.0;      // m

    bool follows = predicted > 0.0 && abs(miss) <= tolerance;
    vec4 before = texelFetch(history, ivec2(floor(texCoord * size)), 0);
    for (int k = 1; k < restart_frames; k++) {
        follows = follows && !(before[k] > 0.0 && abs(center - before[k]) > tolerance);
    }
    vec2 emas = follows ? last + ema * (vec2(miss, abs(miss)) - last) : vec2(0.0);
    if (catch_up_pass) {
        fragColor = vec4(emas, 0.0, 1.0);
        return;
    }
    if (!follows) {
        fragColor = vec4(0.0, measurement, 0.0, 1.0 / measurement);
        return;
    }
    float full = alpha;
    if (catching) {
        float opened = catch_up_share(emas.x, emas.y) * float(warp && moving(size));
        float follow_time = smoothing + opened * (catch_up_smoothing - smoothing);
        full = smoothing > 0.0 ? 1.0 - exp(-delta_time * 1000.0 / max(follow_time, 1e-6)) : 1.0;
    }
    float a = full * share;
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
    fragColor = vec4(z * slope_weight(center), disparity, rate, anchor);
}
