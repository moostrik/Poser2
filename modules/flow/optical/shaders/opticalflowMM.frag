#version 460 core

// Optical Flow Shader with multi-scale detection using explicit offsets

uniform sampler2D tex0;
uniform sampler2D tex1;

uniform vec2 offset;
uniform float threshold;
uniform vec2 force;
uniform float power;

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.0001

vec2 compute_flow_at_offset(vec2 st, vec2 off) {
    vec2 off_x = vec2(off.x, 0.0);
    vec2 off_y = vec2(0.0, off.y);

    float scr_dif = texture(tex0, st).x - texture(tex1, st).x;

    float gradx = texture(tex1, st + off_x).x - texture(tex1, st - off_x).x;
    gradx += texture(tex0, st + off_x).x - texture(tex0, st - off_x).x;

    float grady = texture(tex1, st + off_y).x - texture(tex1, st - off_y).x;
    grady += texture(tex0, st + off_y).x - texture(tex0, st - off_y).x;

    float gradmag = sqrt(gradx * gradx + grady * grady + TINY);

    vec2 flow;
    flow.x = scr_dif * (gradx / gradmag);
    flow.y = scr_dif * (grady / gradmag);

    return flow * force;
}

void main() {
    vec2 st = texCoord;

    // Multi-scale with explicit offsets (clear and controllable)
    vec2 flow_fine = compute_flow_at_offset(st, offset);   // Small motions
    vec2 flow_medium = compute_flow_at_offset(st, offset * 2.0);       // Medium motions
    vec2 flow_coarse = compute_flow_at_offset(st, offset * 4.0); // Large motions

    float mag_fine = length(flow_fine);
    float mag_medium = length(flow_medium);
    float mag_coarse = length(flow_coarse);

    // Pick strongest
    vec2 flow;
    if (mag_coarse > mag_medium && mag_coarse > mag_fine) {
        flow = flow_coarse;
    } else if (mag_medium > mag_fine) {
        flow = flow_medium;
    } else {
        flow = flow_fine;
    }

    // flow = flow_fine;

    // Threshold and normalize
    float magnitude = length(flow);
    magnitude = max(magnitude, threshold);
    magnitude -= threshold;
    magnitude /= (1.0 - threshold + TINY);
    magnitude = pow(magnitude, power);

    flow += TINY;
    flow = normalize(flow) * clamp(magnitude, 0.0, 1.0);

    fragColor = vec4(flow, 0.0, 1.0);
}