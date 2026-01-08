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

    // Choose background color based on joint index
    vec4 bg_color = (joint_index % 2 == 0) ? color_even : color_odd;

    if (joint_index >= num_joints) {
        fragColor = bg_color;
        return;
    }

    float value = texelFetch(values_buffer, joint_index).r;
    float score = texelFetch(scores_buffer, joint_index).r;

    if (score <= 0.0) {
        fragColor = bg_color;
        return;
    }

    float normalized_value = (value - value_min) / (value_max - value_min);
    normalized_value = clamp(normalized_value, 0.0, 1.0);

    // Draw a thick horizontal line at normalized_value with smooth edges inward

    float dist = abs(texCoord.y - normalized_value);
    float alpha = 1.0 - smoothstep(line_thickness, line_thickness + line_smooth, dist);

    if (alpha > 0.01) {
        vec3 line_color = bg_color.rgb;
        // Blend line color and bg_color using alpha for smooth transition
        vec3 blended = mix(bg_color.rgb, line_color, alpha);
        fragColor = vec4(blended, mix(bg_color.a, color.a, alpha));
    } else {
        fragColor = bg_color;
    }

}