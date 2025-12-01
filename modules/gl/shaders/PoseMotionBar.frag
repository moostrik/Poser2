#version 460 core

uniform int num_joints;
uniform float value_min;
uniform float value_max;
uniform float line_thickness = 0.001;
uniform float line_smooth = 0.001;

uniform samplerBuffer values_buffer;
uniform samplerBuffer scores_buffer;

// Add these uniforms for color control
uniform vec4 color;
uniform vec4 color_odd;
uniform vec4 color_even;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float joint_width = 1.0 / float(num_joints);
    int joint_index = int(texCoord.x / joint_width);

    if (joint_index >= num_joints) {
        discard;
    }

    float value = texelFetch(values_buffer, joint_index).r;
    float score = texelFetch(scores_buffer, joint_index).r;

    // if (score <= 0.0) {
    //     discard;
    // }

    float normalized_value = (value - value_min) / (value_max - value_min);
    normalized_value = clamp(normalized_value, 0.0, 1.0);

    // Compute half-height of the filled region
    float half_fill = 0.5 * normalized_value;

    // Centered region: [0.5 - half_fill, 0.5 + half_fill]
    float dist_to_fill = max(abs(texCoord.y - 0.5) - half_fill, 0.0);

    // Smooth edge at the border of the filled region
    float alpha = 1.0 - smoothstep(0.0, line_smooth, dist_to_fill);

    if (alpha > 0.01) {
        fragColor = vec4(color.rgb, color.a * alpha);
    } else {
        discard;
    }

}