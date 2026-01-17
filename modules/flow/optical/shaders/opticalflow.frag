#version 460 core

// Optical Flow Shader
// Ported from ofxFlowTools ftOpticalFlowShader.h
// Computes Lucas-Kanade style sparse optical flow

uniform sampler2D tex0;  // Current frame (luminance)
uniform sampler2D tex1;  // Previous frame (luminance)

uniform vec2 offset;     // Gradient sample offset (normalized)
uniform float threshold; // Motion threshold
uniform vec2 force;      // Force/strength multiplier (includes inversion)
uniform float power;     // Power curve for magnitude

in vec2 texCoord;
out vec4 fragColor;

#define TINY 0.0001

void main() {
    vec2 st = texCoord;
    vec2 off_x = vec2(offset.x, 0.0);
    vec2 off_y = vec2(0.0, offset.y);

    // Get temporal difference (brightness change)
    float scr_dif = texture(tex0, st).x - texture(tex1, st).x;

    // Calculate spatial gradient using both frames
    float gradx, grady, gradmag;

    gradx  = texture(tex1, st + off_x).x - texture(tex1, st - off_x).x;
    gradx += texture(tex0, st + off_x).x - texture(tex0, st - off_x).x;

    grady  = texture(tex1, st + off_y).x - texture(tex1, st - off_y).x;
    grady += texture(tex0, st + off_y).x - texture(tex0, st - off_y).x;

    gradmag = sqrt(gradx * gradx + grady * grady + TINY);

    // Compute flow using brightness constancy assumption
    // (Lucas-Kanade: flow = -It/|grad I|)
    vec2 flow;
    flow.x = scr_dif * (gradx / gradmag);
    flow.y = scr_dif * (grady / gradmag);

    // Apply force (for normalization and inversion)
    flow *= force;

    // Apply threshold and power curve
    float magnitude = length(flow);
    magnitude = max(magnitude - threshold, 0.0);  // Simpler threshold
    magnitude = pow(magnitude, power);

    // Scale flow by new magnitude, then clamp
    flow = (length(flow) > TINY) ? (flow / length(flow)) * magnitude : vec2(0.0);
    flow = clamp(flow, vec2(-1.0), vec2(1.0));


    // Output: RG = velocity XY, BA unused
    fragColor = vec4(flow, 0.0, 1.0);
}
