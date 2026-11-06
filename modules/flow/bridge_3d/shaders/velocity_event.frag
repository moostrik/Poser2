#version 460 core

// Velocity Event Shader
// One event of the motion, peak-held: fast - slow along the motion's direction, its rise (a start,
// sign +1) or its fall (a stop, sign -1). An event counts as far as its patch changes the same way
// (the gate, strength 0 = every event counts); a limb arriving at or leaving a pixel changes that pixel,
// not its patch. A start reads the patch's mean x/y velocity, in which jitter cancels, against a floor
// so a still patch reads no speedup; a stop its mean speed, which jitter never lowers, and which a
// turning limb keeps. The numpy references are velocity_event_bridge.velocity_event and event_gate

uniform sampler2D tex0;  // Fast trail RGB
uniform sampler2D tex1;  // Slow trail RGB
uniform sampler2D tex2;  // Previous held event R
uniform sampler2D tex3;  // Patch means, by mip level: for a start fast x/y RG, slow x/y BA; for a stop fast speed R, slow speed G
uniform float sign;      // +1 start, -1 stop
uniform float decay;     // What remains of the held event this update
uniform float strength;  // Gate strength, 0 = every event counts
uniform float lod;       // Mip level of a patch

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;
const float START_FLOOR = 0.01;     // widths/s: a patch slower than this has no speedup; START_FLOOR in velocity_event_bridge

void main() {
    vec3 fast = texture(tex0, texCoord).xyz;
    vec3 slow = texture(tex1, texCoord).xyz;

    // A stopped fast trail has no direction left; the slow trail still holds it
    float fast_length = length(fast);
    float slow_length = length(slow);
    vec3 unit = fast_length > EPSILON ? fast / fast_length : slow_length > EPSILON ? slow / slow_length : vec3(0.0);

    float event = max(0.0, sign * dot(fast - slow, unit));
    if (strength > 0.0) {
        vec4 patch_mean = textureLod(tex3, texCoord, lod);
        float change;
        if (sign > 0.0) {
            float patch_fast = length(patch_mean.rg);
            change = (patch_fast - length(patch_mean.ba)) / max(patch_fast, START_FLOOR);
        } else {
            change = (patch_mean.g - patch_mean.r) / max(patch_mean.g, EPSILON);
        }
        event *= mix(1.0, clamp(change, 0.0, 1.0), strength);
    }
    float held = texture(tex2, texCoord).r * decay;
    fragColor = vec4(max(event, held), 0.0, 0.0, 1.0);
}
