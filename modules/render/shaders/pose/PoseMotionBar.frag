#version 460 core

uniform int num_joints;
uniform float value_min;
uniform float value_max;
uniform float line_thickness = .001;
uniform float line_smooth = 0.1;

uniform float values[32];
uniform float scores[32];

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

    float value = values[joint_index];
    float score = scores[joint_index];

    float normalized_value = (value - value_min) / (value_max - value_min);
    normalized_value = clamp(normalized_value, 0.0, 1.0);

    // Compute half-height of the filled region
    float half_fill = 0.5 * normalized_value;

    // Check if we're inside the vertical fill range first
    float dist_from_center_y = abs(texCoord.y - 0.5);

    if (dist_from_center_y > half_fill) {
        discard;
    }

    // Now apply horizontal taper based on vertical position
    // At center (y=0.5), full width. At edges (y=0 or 1), narrow width
    float taper_amount = 0.97; // Adjust this: 0.0 = no taper, 1.0 = full taper to point
    float vertical_position = dist_from_center_y / half_fill; // 0 at center, 1 at fill edge

    // Use power function for teardrop shape instead of linear
    float taper_curve = pow(vertical_position, 1.0 - pow(vertical_position, 0.2)); // Adjust exponent for desired curve

    float max_half_width = joint_width * 0.5;
    float tapered_half_width = max_half_width * (1.0 - taper_amount * taper_curve);

    // Distance from center of bar in X direction
    float local_x = mod(texCoord.x, joint_width);
    float dist_from_bar_center = abs(local_x - joint_width * 0.5);

    if (dist_from_bar_center > tapered_half_width) {
        discard;
    }

    // Smooth the horizontal edges
    float edge_dist = tapered_half_width - dist_from_bar_center;
    float alpha = smoothstep(0.0, 0.002, edge_dist);

    fragColor = vec4(color.rgb, color.a * alpha);
}