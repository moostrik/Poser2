#version 460 core

uniform sampler2D tex0;
uniform float inv_width;
uniform int num_streams;
uniform float stream_range;
uniform float line_width;

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec2 uv = vec2(texCoord.x, 1.0 - texCoord.y);

    // Determine which stream we're in
    int stream_idx = int(floor(uv.y / stream_range));
    stream_idx = clamp(stream_idx, 0, num_streams - 1);

    // Get normalized y position within the stream
    float stream_y = (uv.y - float(stream_idx) * stream_range) / stream_range;

    // Sample current r value
    vec3 current_sample = texture(tex0, vec2(uv.x, float(stream_idx) / float(num_streams))).rgb;
    float current_r = 1.0 - current_sample.r;
    float current_valid = current_sample.g;

    // Early exit if invalid
    if (current_valid < 0.5) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Sample previous r value for line interpolation
    vec2 prev_uv = vec2(max(0.0, uv.x - inv_width), uv.y);
    vec3 prev_sample = texture(tex0, vec2(prev_uv.x, float(stream_idx) / float(num_streams))).rgb;
    float prev_r = 1.0 - prev_sample.r;
    float prev_valid = prev_sample.g;

    // If we don't have previous data, just draw a point
    float target_r = current_r;
    float prev_target_r = current_r;

    if (prev_valid > 0.5 && uv.x > inv_width) {
        // Use actual previous value
        prev_target_r = prev_r;
        // Interpolate between previous and current
        float t = fract(uv.x / inv_width);
        target_r = mix(prev_r, current_r, t);
    }

    // Use squared distance method from WS_PoseStream
    vec2 current_point = vec2(uv.x, target_r);
    vec2 prev_point = vec2(uv.x - inv_width, prev_target_r);

    // Optimized distance calculation
    vec2 line_dir = current_point - prev_point;
    vec2 to_pixel = vec2(uv.x, stream_y) - prev_point;

    // Use squared distance comparison to avoid sqrt
    float line_length_sq = dot(line_dir, line_dir);
    float cross_product = line_dir.x * to_pixel.y - line_dir.y * to_pixel.x;
    float dist_to_line_sq = (cross_product * cross_product) / line_length_sq;

    // Convert line_width to squared distance (0.05^2 = 0.0025)
    float line_width_sq = 0.05 * 0.05;

    // Background color: dark grey for odd, black for even streams
    vec3 bg_color = (stream_idx % 2 == 0) ? vec3(0.0) : vec3(0.2);

    // Line color is always white
    vec3 line_color = vec3(1.0);

    // Use squared distance for line rendering
    if (dist_to_line_sq > line_width) {
        fragColor = vec4(bg_color, 1.0);
        if (current_valid > 0.5) {
            fragColor.g += 0.2; // Slight green tint where data exists
        }
    } else {
        fragColor = vec4(line_color, 1.0);
    }
}