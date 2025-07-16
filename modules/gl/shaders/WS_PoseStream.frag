#version 460 core

uniform sampler2D tex0;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 uv = texCoord;
    uv.y = 1.0 - uv.y; // Flip the Y coordinate
    // vec2 prev_uv = uv - vec2(0.0, 1.0 / float(textureSize(tex0, 0).y)); // Offset to the previous pixel in the y-direction

    int num_joints = textureSize(tex0, 0).y;

    // Get the texture data (angles_norm, sign_channel, confidences)
    vec3 data = texture(tex0, uv).rgb;
    float angles_norm = data.r;
    float sign_channel = data.g;
    float confidences = data.b;
    // Calculate the difference between current pixel and previous pixel in x-direction (same joint)
    vec2 prev_uv = uv - vec2(1.0 / float(textureSize(tex0, 0).x), 0.0); // Offset to the previous pixel in the x-direction
    float prev_angles_norm = texture(tex0, prev_uv).r;

    // Calculate joint ID - each row is one joint
    int joint_id = int(floor(uv.y * float(num_joints)));

    // Calculate positions for current and previous points
    float joint_range = 1.0 / float(num_joints + 2);
    float joint_center = (float(joint_id) + 0.5) / float(num_joints) - 0.5 * joint_range;

    // Current point
    vec2 current_point = vec2(uv.x, joint_center + angles_norm * joint_range);
    // Previous point
    vec2 prev_point = vec2(uv.x - 1.0 / float(textureSize(tex0, 0).x), joint_center + prev_angles_norm * joint_range);

    // Calculate distance from current pixel to the line segment
    vec2 line_dir = current_point - prev_point;
    vec2 to_pixel = vec2(uv.x, uv.y) - prev_point;
    float line_length = length(line_dir);
    float dist_to_line = abs(cross(vec3(line_dir, 0.0), vec3(to_pixel, 0.0)).z) / line_length;

    // Dynamic line thickness based on angle change
    float delta_y = abs(angles_norm - prev_angles_norm);
    float base_thickness = 0.005;

    // Hard cutoff for line alpha
    float line_alpha = dist_to_line < base_thickness ? 1.0 : 0.0;

    // Early exit if alpha is negligible
    if (line_alpha < 0.001) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Color based on joint (even/odd) and angle sign
    vec3 color;
    bool is_even = (joint_id % 2 == 0);
    bool is_positive = (sign_channel > 0.5);

    if (is_even) {
        if (is_positive) {
            color = vec3(1.0, 1.0, 0.0); // Yellow
        } else {
            color = vec3(1.0, 0.0, 0.0); // Red
        }
    } else {
        if (is_positive) {
            color = vec3(0.0, 0.7, 1.0); // Blue
        } else {
            color = vec3(0.0, 1.0, 0.0); // Green
        }
    }

    // Apply confidence and line alpha
    float alpha = confidences * line_alpha;

    fragColor = vec4(color, alpha);
}