#version 460 core

uniform int num_joints;
uniform int num_colors;
uniform float value_min;
uniform float value_max;
uniform float line_thickness = 0.01;
uniform float line_smooth = 0.01;
uniform int use_scores = 0;

uniform float values[32];
uniform float scores[32];
uniform vec4 colors[8];  // Cycle through these colors

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float joint_width = 1.0 / float(num_joints);
    int joint_index = int(texCoord.x / joint_width);

    if (joint_index >= num_joints) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float value = values[joint_index];
    float score = use_scores != 0 ? scores[joint_index] : 1.0;

    if (use_scores != 0 && score <= 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Get color by cycling through color array
    int color_index = joint_index % num_colors;
    vec4 line_color = colors[color_index];

    // Normalize value to [0, 1]
    float normalized_value = (value - value_min) / (value_max - value_min);
    normalized_value = clamp(normalized_value, 0.0, 1.0);

    // Calculate distance from horizontal line
    float dist = abs(texCoord.y - normalized_value);

    // Smooth antialiased line
    float alpha = 1.0 - smoothstep(line_thickness, line_thickness + line_smooth, dist);

    if (alpha > 0.01) {
        fragColor = vec4(line_color.rgb, line_color.a * alpha * score);
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
}
