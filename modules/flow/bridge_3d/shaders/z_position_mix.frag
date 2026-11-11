#version 460 core

// Z Position Mix
// An output's z position: its readings' z positions weighted by each reading's size at its share, each times
// the share of its trail's speed that came with a z position (the trail's z position weight over its speed),
// so motion without depth does not pull it. The trail reads the lowpass's z position, a start and a bow the
// fast trail's, a stop and a wake the slow trail's, the change the fast trail's where it is the faster, else
// the slow's. Out: red the z position times its weight, green the weight. The numpy reference is
// bridge_3d.z_position_mix

uniform sampler2D tex0;     // the readings: trail, change, start, bow, stop, wake (xyz)
uniform sampler2D tex1;
uniform sampler2D tex2;
uniform sampler2D tex3;
uniform sampler2D tex4;
uniform sampler2D tex5;
uniform float shares[6];
uniform sampler2D trail;            // the lowpass's trail (xyz)
uniform sampler2D fast;             // the bandpass's fast trail
uniform sampler2D slow;             // its slow trail
uniform sampler2D trail_z_position; // their z positions (red over green)
uniform sampler2D fast_z_position;
uniform sampler2D slow_z_position;

in vec2 texCoord;
out vec4 fragColor;

const float EPSILON = 1e-4;     // EPSILON in trails: below this a speed is no motion

// A reading's z position at its trail, red over green, per unit of the trail's speed
vec2 per_speed(sampler2D z_position, float speed) {
    return speed > EPSILON ? texture(z_position, texCoord).rg / speed : vec2(0.0);
}

float size(sampler2D tex, float share) {
    return share > 0.0 ? share * length(texture(tex, texCoord).xyz) : 0.0;
}

void main() {
    float trail_speed = length(texture(trail, texCoord).xyz);
    float fast_speed = length(texture(fast, texCoord).xyz);
    float slow_speed = length(texture(slow, texCoord).xyz);
    vec2 at_trail = per_speed(trail_z_position, trail_speed);
    vec2 at_fast = per_speed(fast_z_position, fast_speed);
    vec2 at_slow = per_speed(slow_z_position, slow_speed);

    vec2 sum = size(tex0, shares[0]) * at_trail
             + size(tex1, shares[1]) * (fast_speed > slow_speed ? at_fast : at_slow)
             + (size(tex2, shares[2]) + size(tex3, shares[3])) * at_fast
             + (size(tex4, shares[4]) + size(tex5, shares[5])) * at_slow;
    fragColor = vec4(sum, 0.0, 0.0);
}
