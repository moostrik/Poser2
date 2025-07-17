#version 460 core

uniform sampler2D   tex0;
uniform float       sample_step;
uniform int         num_streams;
uniform float       stream_step;
uniform float       line_width;

in vec2     texCoord;
out vec4    fragColor;

void main() {
    vec2    uv = vec2(texCoord.x, 1.0 - texCoord.y);

    // Calculate stream ID
    int     stream_id = int(floor(uv.y / stream_step));

    // Calculate stream center
    float   stream_center = (float(stream_id) + 0.5) * stream_step;

    // Get current and previous UV coordinates
    vec2    curr_uv = vec2(texCoord.x, stream_center);
    vec2    prev_uv = vec2(max(0.0, texCoord.x - sample_step), stream_center);

    // Get curr sample and its properties
    vec3    curr_sample = texture(tex0, curr_uv).rgb;
    float   curr_value = 1.0 - curr_sample.r;
    float   curr_valid = curr_sample.g;

    // Early exit if invalid
    if (curr_valid < 0.5) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Get previous sample and its properties
    vec3    prev_sample = texture(tex0, prev_uv).rgb;
    float   prev_value = 1.0 - prev_sample.r;
    float   prev_valid = prev_sample.g;

    // Early exit if invalid
    if (prev_valid < 0.5) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Calculate stream center
    float   stream_top = stream_center - 0.5 * stream_step;

    // Current and previous points
    vec2    curr_point = vec2(curr_uv.x, stream_top + curr_value * stream_step);
    vec2    prev_point = vec2(prev_uv.x, stream_top + prev_value * stream_step);

    // Optimized distance calculation
    vec2    line_dir = curr_point - prev_point;
    vec2    to_pixel = vec2(uv.x, uv.y) - prev_point;

    // Use squared distance comparison to avoid sqrt
    float   line_length_sq =  dot(line_dir, line_dir);
    float   cross_product =   line_dir.x * to_pixel.y - line_dir.y * to_pixel.x;
    float   dist_to_line_sq = (cross_product * cross_product) / line_length_sq;

    // Background color: dark grey for odd, black for even streams
    vec3 bg_color = (stream_id % 2 == 0) ? vec3(0.0) : vec3(0.2);

    // Line color is always white
    vec3 line_color = vec3(1.0);

    // Use squared distance for line rendering
    if (dist_to_line_sq > line_width) {
        fragColor = vec4(bg_color, 1.0);
        if (curr_valid > 0.5) {
            fragColor.g += 0.2; // Slight green tint where data exists
        }
    } else {
        fragColor = vec4(line_color, 1.0);
    }
}