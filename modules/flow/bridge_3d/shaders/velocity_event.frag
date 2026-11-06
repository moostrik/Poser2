#version 460 core

// Velocity Event Shader
// One event of the motion, peak-held: fast - slow along the motion's direction, its rise (a start,
// sign +1) or its fall (a stop, sign -1). It counts as far as its region changes the same way, by the
// region weight (0 = every pixel's event counts); a limb arriving at or leaving a pixel changes that
// pixel, not its region.
// A start reads the region's mean velocity over its moving pixels, smoothed per region by the fast and
// the slow trail: a limb arriving keeps that mean, a limb speeding up raises it, jitter on a still
// region moves none of its pixels, and a region slower than the floor has no speedup. A stop reads the
// region's mean speed, which jitter never lowers and a turning limb keeps.
// The numpy references are velocity_event_bridge.velocity_event, start_region_share and stop_region_share

uniform sampler2D tex0;         // Fast trail RGB
uniform sampler2D tex1;         // Slow trail RGB
uniform sampler2D tex2;         // Previous held event R
uniform sampler2D tex3;         // Start: the region's mean moving velocity, fast trail RGB; stop: region means, fast speed R, slow speed G, by mip level
uniform sampler2D tex4;         // Start: the region's mean moving velocity, slow trail RGB; stop: unread
uniform float sign;             // +1 start, -1 stop
uniform float decay;            // What remains of the held event this update
uniform float region_weight;    // 0 = every pixel's event counts, 1 = as far as its region changes the same way
uniform float lod;              // Stop: mip level of a region

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;
const float START_FLOOR = 0.01;     // widths/s: a region slower than this has no speedup; START_FLOOR in velocity_event_bridge

void main() {
    vec3 fast = texture(tex0, texCoord).xyz;
    vec3 slow = texture(tex1, texCoord).xyz;

    // A stopped fast trail has no direction left; the slow trail still holds it
    float fast_length = length(fast);
    float slow_length = length(slow);
    vec3 unit = fast_length > EPSILON ? fast / fast_length : slow_length > EPSILON ? slow / slow_length : vec3(0.0);

    float event = max(0.0, sign * dot(fast - slow, unit));
    if (region_weight > 0.0) {
        float change;
        if (sign > 0.0) {
            float region_fast = length(texture(tex3, texCoord).xyz);
            change = (region_fast - length(texture(tex4, texCoord).xyz)) / max(region_fast, START_FLOOR);
        } else {
            vec2 region_speed = textureLod(tex3, texCoord, lod).rg;     // fast, slow
            change = (region_speed.g - region_speed.r) / max(region_speed.g, EPSILON);
        }
        event *= mix(1.0, clamp(change, 0.0, 1.0), region_weight);
    }
    float held = texture(tex2, texCoord).r * decay;
    fragColor = vec4(max(event, held), 0.0, 0.0, 1.0);
}
