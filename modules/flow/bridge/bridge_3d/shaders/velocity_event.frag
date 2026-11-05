#version 460 core

// Velocity Event Shader
// One event of the motion, peak-held: fast - slow along the motion's direction, its rise (a start,
// sign +1) or its fall (a stop, sign -1). A stop counts as far as its patch is slowing down (the stop
// gate, strength 0 = every stop counts). The numpy reference is velocity_event_bridge.velocity_event

uniform sampler2D tex0;  // Fast trail RGB
uniform sampler2D tex1;  // Slow trail RGB
uniform sampler2D tex2;  // Previous held event R
uniform sampler2D tex3;  // Patch means: fast speed R, slow speed G, by mip level
uniform float sign;      // +1 start, -1 stop
uniform float decay;     // What remains of the held event this update
uniform float strength;  // Gate strength, 0 = every event counts
uniform float lod;       // Mip level of a patch

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;

void main() {
    vec3 fast = texture(tex0, texCoord).xyz;
    vec3 slow = texture(tex1, texCoord).xyz;

    // A stopped fast trail has no direction left; the slow trail still holds it
    float fast_length = length(fast);
    float slow_length = length(slow);
    vec3 unit = fast_length > EPSILON ? fast / fast_length : slow_length > EPSILON ? slow / slow_length : vec3(0.0);

    float event = max(0.0, sign * dot(fast - slow, unit));
    if (strength > 0.0) {
        vec2 patch_speed = textureLod(tex3, texCoord, lod).rg;
        float slowdown = clamp((patch_speed.g - patch_speed.r) / max(patch_speed.g, EPSILON), 0.0, 1.0);
        event *= mix(1.0, slowdown, strength);
    }
    float held = texture(tex2, texCoord).r * decay;
    fragColor = vec4(max(event, held), 0.0, 0.0, 1.0);
}
