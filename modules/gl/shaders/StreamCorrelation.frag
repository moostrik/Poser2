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

    // Draw line between previous and current point
    vec2 line_vector = curr_point - prev_point;
    float line_length = length(line_vector);

    // Normalize the line vector
    vec2 line_dir = line_vector / line_length;

    // Vector from prev_point to current fragment
    vec2 to_fragment = uv - prev_point;

    // Project the vector onto the line
    float projection = dot(to_fragment, line_dir);
    projection = clamp(projection, 0.0, line_length);

    // Find the closest point on the line segment
    vec2 closest_point = prev_point + projection * line_dir;

    // Distance from fragment to the line
    float dist = length(uv - closest_point);

    // Set color based on distance (white if close enough, bg_color otherwise)
    vec3 bg_color = ((stream_id & 1) == 0) ? vec3(0.15) : vec3(0.25);
    if (dist <= line_width) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0); // White line
    } else {
        fragColor = vec4(bg_color, 1.0); // Background
    }
}
