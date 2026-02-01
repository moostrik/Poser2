#version 460 core

uniform sampler2D   tex0;
uniform float       sample_step;
uniform int         num_streams;
uniform float       stream_step;
uniform float       line_width;
uniform float       output_aspect_ratio;  // output buffer aspect ratio (width/height)
uniform float       display_range_min;   // minimum value for display range
uniform float       display_range_max;   // maximum value for display range
uniform vec3        color_even;          // color for even-numbered streams
uniform vec3        color_odd;           // color for odd-numbered streams
uniform float       alpha;               // alpha transparency

in vec2     texCoord;
out vec4    fragColor;


bool isBetweenStreamLine(vec2 uv, float stream_step, float line_width, int num_streams) {
    int stream_id = int(floor(uv.y / stream_step));
    if (stream_id >= num_streams - 1) return false;

    float boundary_y = (float(stream_id + 1)) * stream_step;
    return abs(uv.y - boundary_y) < line_width;
}

void main() {
    vec2 stream_uv = vec2(texCoord.x, 1.0 - texCoord.y);
    vec2 frag_pos = vec2(texCoord.x, stream_uv.y);

    // Draw stream separator lines
    if (isBetweenStreamLine(stream_uv, stream_step, line_width / output_aspect_ratio * 0.5, num_streams)) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    // Calculate stream ID and bounds
    int stream_id = int(floor(stream_uv.y / stream_step));
    float stream_center = (float(stream_id) + 0.5) * stream_step;
    float stream_top = stream_center - 0.5 * stream_step;

    // Sample current and previous directly from texture
    vec2 value_uv = vec2(texCoord.x, stream_center);
    float value = texture(tex0, value_uv).r;
    float mask = texture(tex0, value_uv).g;

    if (mask < 0.001) {
        fragColor = vec4(0.0);
        return;
    }

    float norm = clamp((value - display_range_min) / (display_range_max - display_range_min), 0.0, 1.0);
    vec2 point = vec2(value_uv.x, stream_top + norm * stream_step);

    // Draw continuous dots (no spacing)
    float dot_radius = line_width * 2.0;
    vec2 diff = frag_pos - point;
    // Correct for aspect ratio by dividing Y
    diff.y /= output_aspect_ratio;
    float dist = length(diff);
    if (dist < dot_radius) {
        vec3 dot_color = (stream_id % 2 == 0) ? color_even : color_odd;
        fragColor = vec4(dot_color, alpha);
        return;
    }

    fragColor = vec4(0.0);
}
