#version 460 core

uniform int num_joints;
uniform float value_min;
uniform float value_max;
uniform samplerBuffer values_buffer;
uniform samplerBuffer scores_buffer;

// Add these uniforms for color control
uniform vec4 color_low;
uniform vec4 color_high;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float joint_width = 1.0 / float(num_joints);
    int joint_index = int(texCoord.x / joint_width);

    if (joint_index >= num_joints) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float value = texelFetch(values_buffer, joint_index).r;
    float score = texelFetch(scores_buffer, joint_index).r;

    if (score <= 0.0) {
        fragColor = vec4(0.2, 0.2, 0.2, 0.0);
        return;
    }

    float normalized_value = (value - value_min) / (value_max - value_min);
    normalized_value = clamp(normalized_value, 0.0, 1.0);

    // Draw a thick horizontal line at normalized_value with smooth edges inward
    float line_thickness = 0.002; // Center thickness
    float edge_smooth = 0.001;    // Smoothing range

    float dist = abs(texCoord.y - normalized_value);
    float alpha = 1.0 - smoothstep(line_thickness - edge_smooth, line_thickness, dist);

    if (alpha > 0.01) {
        vec3 color = mix(color_low.xyz, color_high.xyz, normalized_value);
        color *= score;
        fragColor = vec4(color, alpha);
    } else {
        fragColor = vec4(0.1, 0.1, 0.1, 0.0);
    }

    // Separator lines between bars
    float bar_x = mod(texCoord.x, joint_width) / joint_width;
    if (bar_x < 0.01 || bar_x > 0.99) {
        fragColor = vec4(0.5, 0.5, 0.5, 1.0);
    }
}