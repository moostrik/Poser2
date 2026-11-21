#version 460 core

// Depth Velocity Shader
// One frame of the depth track, in disparity (1 / depth): the track the frame before, carried from
// where the pixel was `displacement` (uv per frame) back, predicted on by its rate and corrected by
// the new measurement, an alpha–beta filter. The measurement is the mean disparity over a window of
// depth samples, of those within `edge` of the pixel's depth (not the background beside the player)
// that the camera measured (the view's filled depth, confidence 0, left out); it corrects by alpha and
// beta times the share of the window it holds, so a thin window corrects little.
// The gate (a tracker's validation gate, Bar-Shalom): the measurement follows the track when it lies
// within the motion a body part makes in the frame, `max_speed` × `delta_time` in disparity, plus 3 σ
// of the window mean's noise; a sample's σ is `depth_noise` over its confidence squared (stereo noise is
// about constant in disparity, and grows as the camera is less sure), the mean's the root of their
// squares' sum over their count. The track is carried only from the texels whose disparity, a frame on
// by its rate, lies within the same gate, so it is never carried across another surface. A measurement
// outside the gate, or none (only filled depth), is a miss: the track predicts on uncorrected; a second
// miss in a row starts it over with no rate (another surface passing over the pixel), as does a filtered
// rate past `max_speed` (a depth sliding away too fast). The misses in a row go to the second target.
// 0 `max_speed` gates nothing. Without depth at the pixel or without a span, it is empty. z is the rate
// over the meters one view width spans there, `depth × view_span`, in view widths per frame. Depth is
// read nearest, so an edge never reads as a distance in between.
// The time the track follows over is the 1€ filter's (Casiez, Roussel, Vogel 2012): `smoothing` at rest,
// over 1 + speed / `halving_speed` in a motion, past `rest_speed`: the speed the larger of x/y's here (its 3×3
// displacement) and the carried track's own z, both px per frame; 0 `halving_speed` is `smoothing` throughout.
// With `dead_zone` above 0, z is 0 while the tracked depth stays within it of its anchor; past it, z at its
// full speed, and the anchor follows at `dead_zone` behind (a backlash); a track that starts over is anchored anew.

uniform sampler2D tex0;  // Depth now R (m), its confidence G (0 to 1, 0 where the view filled it)
uniform sampler2D tex1;  // The track the frame before: z, disparity (1/m), rate (1/m per second), anchor (m)
uniform sampler2D misses;           // The track's misses in a row the frame before
uniform sampler2D displacement;     // uv per frame (RG), with warp
uniform bool warp;
uniform vec2 depth_step; // uv between two depth samples
uniform int window;      // Radius in depth samples
uniform float edge;      // Depth step to another surface within a frame (m)
uniform float smoothing;   // Time the track follows over at rest (ms), `smooth` in Python (`smooth` is a GLSL keyword); 0 takes the measurement whole
uniform float halving_speed;       // speed past rest_speed at which that time is half (px per frame); 0 off
uniform float rest_speed;          // speed below which the motion is at rest (px per frame)
uniform float delta_time;           // Seconds since the last depth
uniform float view_span; // Meters one view width spans per meter of depth
uniform float dead_zone; // Depth the track wanders within its anchor without z (m); 0 is none
uniform float max_speed; // The fastest a body part moves in depth (m/s): the gate's motion; faster is another surface; 0 is no gate
uniform float depth_noise;          // The camera's depth noise at 1 m and full confidence (m), which is its disparity noise (1/m)

const float GATE_SIGMAS = 3.0;      // the gate's width in the measurement's σ, the textbook's
const float RESTART_MISSES = 2.0;   // misses in a row that start the track over ("M of N")

in vec2 texCoord;
layout(location = 0) out vec4 fragColor;
layout(location = 1) out float misses_out;

// The guard (FiniteGuard): a value whose exponent bits are all ones is an infinity or a NaN; it becomes 0 and is counted
layout(std430, binding = 7) buffer FiniteGuard { uint uNonFinite[]; };
uniform int uGuardSlot;  // The output's slot in the motion's FiniteGuard; -1 no guard

vec4 guard(vec4 value) {
    if (uGuardSlot < 0 || !any(equal(floatBitsToUint(value) & 0x7F800000u, uvec4(0x7F800000u)))) return value;
    atomicAdd(uNonFinite[uGuardSlot], 1u);
    return vec4(0.0);
}

// The depth and its confidence at uv
vec2 sample_at(vec2 uv) {
    ivec2 size = textureSize(tex0, 0);
    ivec2 texel = clamp(ivec2(floor(uv * vec2(size))), ivec2(0), size - 1);
    return texelFetch(tex0, texel, 0).rg;
}

// The track's disparity, rate and anchor (xyz) at `position` (texels, centers at integers), of the
// texels that hold one of the `reference` disparity's surface, within `gate` (1/m) of it, each taken
// `ahead` seconds on by its rate. None of them: the nearest texel's track, which the gate then counts a
// miss; none there either: no track
vec3 carried(vec2 position, float reference, float ahead, float gate) {
    ivec2 size = textureSize(tex1, 0);
    ivec2 base = ivec2(floor(position));
    vec2 f = position - vec2(base);
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            ivec2 at = clamp(base + ivec2(dx, dy), ivec2(0), size - 1);
            vec4 texel = texelFetch(tex1, at, 0);
            bool same = texel.g > 0.0 && abs(texel.g + texel.b * ahead - reference) <= gate;
            float weight = (dx == 0 ? 1.0 - f.x : f.x) * (dy == 0 ? 1.0 - f.y : f.y) * float(same);
            sum += weight * texel.gba;
            total += weight;
        }
    }
    if (total > 1e-9) return sum / total;
    return texelFetch(tex1, clamp(ivec2(floor(position + 0.5)), ivec2(0), size - 1), 0).gba;
}

// x/y's speed here: the largest displacement of the 3×3 texels (px per frame)
float xy_speed(vec2 size) {
    ivec2 at = ivec2(floor(texCoord * size));
    ivec2 last_texel = ivec2(size) - 1;
    float largest = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 d = texelFetch(displacement, clamp(at + ivec2(i, j), ivec2(0), last_texel), 0).xy * size;
            largest = max(largest, length(d));
        }
    }
    return largest;
}

// A track that starts over: from the measurement with no rate, or empty without one
void start_over(bool measured, float measurement) {
    fragColor = measured ? vec4(0.0, measurement, 0.0, 1.0 / measurement) : vec4(0.0);
    misses_out = 0.0;
}

// One frame of the track into fragColor, its misses in a row into misses_out
void step_track() {
    float center = sample_at(texCoord).r;
    if (center <= 0.0 || view_span <= 0.0) {
        fragColor = vec4(0.0);
        misses_out = 0.0;
        return;
    }

    float total = 0.0;
    float noise_squared = 0.0;
    int kept = 0;
    for (int j = -window; j <= window; j++) {
        for (int i = -window; i <= window; i++) {
            vec2 depth_sample = sample_at(texCoord + vec2(i, j) * depth_step);
            if (depth_sample.r > 0.0 && depth_sample.g > 0.0 && abs(depth_sample.r - center) <= edge) {
                float sigma = depth_noise / (depth_sample.g * depth_sample.g);
                total += 1.0 / depth_sample.r;
                noise_squared += sigma * sigma;
                kept++;
            }
        }
    }
    bool measured = kept > 0;
    float side = float(2 * window + 1);
    float measurement = measured ? total / float(kept) : 0.0;
    float share = float(kept) / (side * side);
    float noise = measured ? sqrt(noise_squared) / float(kept) : depth_noise;   // the mean's σ (1/m)

    // The gate (1/m): the motion a body part makes in the frame, as disparity at this depth, plus the noise
    float reference = measured ? measurement : 1.0 / center;
    float gate = max_speed > 0.0 ? max_speed * delta_time * reference * reference + GATE_SIGMAS * noise : 1e9;

    bool smoothed = smoothing > 0.0;
    vec2 moved = warp ? texture(displacement, texCoord).xy : vec2(0.0);
    vec2 size = vec2(textureSize(tex1, 0));
    vec2 position = texCoord * size - 0.5 - moved * size;
    vec3 track = carried(position, reference, smoothed ? delta_time : 0.0, gate);
    float predicted = track.x > 0.0 ? track.x + track.y * delta_time : 0.0;
    if (predicted <= 0.0) {
        start_over(measured, measurement);
        return;
    }
    ivec2 came_from = clamp(ivec2(floor(position + 0.5)), ivec2(0), ivec2(size) - 1);
    float missed = texelFetch(misses, came_from, 0).r;
    bool follows = measured && abs(measurement - predicted) <= gate;
    if (!follows && missed + 1.0 >= RESTART_MISSES) {
        start_over(measured, measurement);
        return;
    }

    float disparity = predicted;                // a miss: predicted on, uncorrected
    float rate = track.y;
    if (follows) {
        // The 1€ filter's time: `smoothing` at rest, shorter as x/y or the track's own z moves
        float full = 1.0;
        if (smoothed) {
            float follow_time = smoothing;
            if (halving_speed > 0.0) {
                float speed = warp ? xy_speed(size) : 0.0;
                speed = max(speed, abs(track.y) * delta_time / (track.x * view_span) * size.x);
                follow_time = smoothing / (1.0 + max(speed - rest_speed, 0.0) / halving_speed);
            }
            full = 1.0 - exp(-delta_time * 1000.0 / follow_time);
        }
        float a = full * share;
        float b = a * a / (2.0 - a);
        float residual = measurement - predicted;
        disparity = predicted + a * residual;
        rate = track.y + b * residual / delta_time;
        // The speed after the filter: a track whose depth slides away faster than a body part moves, each
        // step within the gate, reads another surface sliding over the pixel (a sloped edge leaving it), not
        // a motion: it starts over. The depth's speed is the disparity's rate over the disparity squared
        if (max_speed > 0.0 && abs(rate) / (disparity * disparity) > max_speed) {
            start_over(measured, measurement);
            return;
        }
    }
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
    misses_out = follows ? 0.0 : missed + 1.0;
}

void main() {
    step_track();
    fragColor = guard(fragColor);
}
