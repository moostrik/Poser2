#version 460 core

uniform sampler2D   tex0;
uniform float       sample_step;
uniform int         num_streams; // redundant, can be
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

    // Get current sample and its properties
    vec3    curr_sample = texture(tex0, curr_uv).rgb;
    float   curr_value = curr_sample.r;
    float   curr_sign = curr_sample.g;
    float   curr_conf = curr_sample.b;

    // Early exit if confidence is too low
    if (curr_conf < 0.001) {
        fragColor = vec4(0.0);
        return;
    }

    // Get previous sample and its properties
    vec3    prev_sample = texture(tex0, prev_uv).rgb;
    float   prev_value = prev_sample.r;
    float   pref_conf = prev_sample.b;

    // Early exit if invalid
    if (pref_conf < 0.001) {
        fragColor = vec4(0.0);
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

    // Compare squared distances (0.005^2 = 0.000025)
    if (dist > line_width) {
        fragColor = vec4(0.0);
        return;
    }

    // Optimized color selection using mix
    bool    is_even = ((stream_id & 1) == 0);
    bool    is_positive = (curr_sign > 0.5);

    // Use conditional assignment instead of nested if statements
    vec3    color = is_even ?
        (is_positive ? vec3(1.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)) :
        (is_positive ? vec3(0.0, 0.7, 1.0) : vec3(0.0, 1.0, 0.0));

    fragColor = vec4(color, curr_conf);
}