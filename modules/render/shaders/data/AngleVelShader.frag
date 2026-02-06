#version 460 core

uniform int num_joints;
uniform float value_min;
uniform float value_max;
uniform float line_thickness = 0.001;
uniform float line_smooth = 0.001;
uniform vec4 color_odd;
uniform vec4 color_even;

uniform float angles[32];
uniform float deltas[32];
uniform float scores[32];

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float joint_width = 1.0 / float(num_joints);
    int joint_index = int(texCoord.x / joint_width);

    // Choose background color based on joint index
    vec4 color = (joint_index % 2 == 0) ? color_even : color_odd;

    float angle = angles[joint_index];
    float delta = deltas[joint_index];
    float thickness = max(abs(delta) * line_thickness * 10, line_thickness);
    float score = 1.0; //scores[joint_index];

    float normalized_value = (angle - value_min) / (value_max - value_min);

    // Handle wrapping - if value exceeds range, wrap it around
    normalized_value = fract(normalized_value);

    float dist = abs(texCoord.y - normalized_value);

    // Also check wrapped distance (for values that wrap around)
    float wrapped_dist = min(dist, min(abs(texCoord.y - (normalized_value + 1.0)),
                                      abs(texCoord.y - (normalized_value - 1.0))));

    float alpha = (1.0 - smoothstep(thickness, thickness + line_smooth, wrapped_dist)) * score;

    fragColor = vec4(color.rgb, color.a * alpha);
}