#version 460 core

// Velocity Change Shader
// One change of the motion: fast - slow along the motion's direction, its rise (sign +1) or its fall
// (sign -1), over the two trails' step peak (`scale`), so a limb at speed v that starts or stops at once
// reads v. Its region splits it in two by the region weight (0 = all of it is the region's): the share
// its region changes the same way is the start or the stop, the rest (`edge` 1) the bow or the wake, a
// limb arriving at or leaving a pixel, which changes that pixel and not its region.
// A rise reads the region's mean velocity over its moving pixels, smoothed per region by the fast and
// the slow trail: a limb arriving keeps that mean, a limb speeding up raises it, and jitter on a still
// region cancels in it. A fall reads the region's mean speed, which jitter never lowers and a turning
// limb keeps.
// The numpy references are velocity_bandpass.velocity_change, start_region_share and stop_region_share

uniform sampler2D tex0;         // Fast trail RGB
uniform sampler2D tex1;         // Slow trail RGB
uniform sampler2D tex2;         // A rise: the region's mean moving velocity, fast trail RGB; a fall: region means, fast speed R, slow speed G, by mip level
uniform sampler2D tex3;         // A rise: the region's mean moving velocity, slow trail RGB; a fall: unread
uniform float sign;             // +1 start or bow, -1 stop or wake
uniform float edge;             // 0 the region's share (start, stop), 1 the rest (bow, wake)
uniform float scale;            // 1 / the trails' step peak; 0 = no changes
uniform float region_weight;    // 0 = every pixel's change counts, 1 = as far as its region changes the same way
uniform float lod;              // A fall: mip level of a region

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;     // EPSILON in trails: below this a speed is no motion

void main() {
    vec3 fast = texture(tex0, texCoord).xyz;
    vec3 slow = texture(tex1, texCoord).xyz;

    // A stopped fast trail has no direction left; the slow trail still holds it
    float fast_length = length(fast);
    float slow_length = length(slow);
    vec3 unit = fast_length > EPSILON ? fast / fast_length : slow_length > EPSILON ? slow / slow_length : vec3(0.0);

    float change = max(0.0, sign * dot(fast - slow, unit)) * scale;
    float share = 1.0;
    if (region_weight > 0.0) {
        float change;
        if (sign > 0.0) {
            float region_fast = length(texture(tex2, texCoord).xyz);
            change = (region_fast - length(texture(tex3, texCoord).xyz)) / max(region_fast, EPSILON);
        } else {
            vec2 region_speed = textureLod(tex2, texCoord, lod).rg;     // fast, slow
            change = (region_speed.g - region_speed.r) / max(region_speed.g, EPSILON);
        }
        share = mix(1.0, clamp(change, 0.0, 1.0), region_weight);
    }
    change *= edge > 0.5 ? 1.0 - share : share;
    fragColor = vec4(change, 0.0, 0.0, 1.0);
}
