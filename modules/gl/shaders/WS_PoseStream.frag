#version 460 core

uniform sampler2D tex0;
uniform float inv_width;
uniform int num_joints;
uniform float joint_range;
uniform float line_width;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 uv = vec2(texCoord.x, 1.0 - texCoord.y);
    vec2 prev_uv = vec2(max(0.0, uv.x - inv_width), uv.y);

    // Get current and previous data in one go
    vec3 data = texture(tex0, uv).rgb;
    float angles_norm = data.r;
    float sign_channel = data.g;
    float confidences = data.b;

    // Early exit if confidence is too low
    if (confidences < 0.001) {
        fragColor = vec4(0.0);
        return;
    }

    float prev_angles_norm = texture(tex0, prev_uv).r;

    // Calculate joint ID
    int joint_id = int(uv.y * float(num_joints));

    // Calculate positions - use precomputed values
    float joint_center = (float(joint_id) + 0.5) / float(num_joints) - 0.5 * joint_range;

    // Current and previous points
    vec2 current_point = vec2(uv.x, joint_center + angles_norm * joint_range);
    vec2 prev_point = vec2(uv.x - inv_width, joint_center + prev_angles_norm * joint_range);

    // Optimized distance calculation
    vec2 line_dir = current_point - prev_point;
    vec2 to_pixel = vec2(uv.x, uv.y) - prev_point;

    // Use squared distance comparison to avoid sqrt
    float line_length_sq = dot(line_dir, line_dir);
    float cross_product = line_dir.x * to_pixel.y - line_dir.y * to_pixel.x;
    float dist_to_line_sq = (cross_product * cross_product) / line_length_sq;

    // Compare squared distances (0.005^2 = 0.000025)
    if (dist_to_line_sq > line_width) {
        fragColor = vec4(0.0);
        return;
    }

    // Optimized color selection using mix
    bool is_even = ((joint_id & 1) == 0);
    bool is_positive = (sign_channel > 0.5);

    // Use conditional assignment instead of nested if statements
    vec3 color = is_even ?
        (is_positive ? vec3(1.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)) :
        (is_positive ? vec3(0.0, 0.7, 1.0) : vec3(0.0, 1.0, 0.0));

    fragColor = vec4(color, confidences);
}