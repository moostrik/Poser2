#version 460 core

uniform int num_joints;
uniform float value_min;
uniform float value_max;
uniform samplerBuffer values_buffer;
uniform samplerBuffer scores_buffer;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    // Calculate which joint we're rendering based on x coordinate
    float joint_width = 1.0 / float(num_joints);
    int joint_index = int(texCoord.x / joint_width);

    // Clamp to valid range
    if (joint_index >= num_joints) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Fetch value and score for this joint
    float value = texelFetch(values_buffer, joint_index).r;
    float score = texelFetch(scores_buffer, joint_index).r;

    // Check if value is valid (not NaN and has confidence)
    if (isnan(value) || score <= 0.0) {
        fragColor = vec4(0.2, 0.2, 0.2, 1.0); // Gray for invalid joints
        return;
    }

    // Normalize value to [0, 1]
    float normalized_value = (value - value_min) / (value_max - value_min);
    normalized_value = clamp(normalized_value, 0.0, 1.0);

    // Draw vertical bar from bottom up to normalized_value height
    if (texCoord.y <= normalized_value) {
        // Color based on value (green = low, yellow = mid, red = high)
        vec3 color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), normalized_value);

        // Dim color based on confidence score
        color *= score;

        fragColor = vec4(color, 1.0);
    } else {
        // Background
        fragColor = vec4(0.1, 0.1, 0.1, 1.0);
    }

    // Add thin white separator lines between bars
    float bar_x = mod(texCoord.x, joint_width) / joint_width;
    if (bar_x < 0.02 || bar_x > 0.98) {
        fragColor = vec4(0.5, 0.5, 0.5, 1.0);
    }
}