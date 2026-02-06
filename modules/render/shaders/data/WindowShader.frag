#version 460 core

uniform sampler2D   tex0;
uniform int         num_streams;
uniform float       stream_step;
uniform float       line_width;
uniform float       output_aspect_ratio;  // output buffer aspect ratio (width/height)
uniform float       display_range_min;   // minimum value for display range
uniform float       display_range_max;   // maximum value for display range
uniform vec4        colors[8];           // RGBA color array to cycle through
uniform int         num_colors;          // number of colors in array
uniform float       alpha;               // alpha transparency

in vec2     texCoord;
out vec4    fragColor;


bool isBetweenStreamLine(vec2 uv, float stream_step, float line_width, int num_streams, float used_height) {
    // Check top boundary (y = 0)
    if (abs(uv.y) < line_width) return true;

    // Check bottom boundary (y = used_height)
    if (abs(uv.y - used_height) < line_width) return true;

    // Check intermediate stream boundaries
    int stream_id = int(floor(uv.y / stream_step));
    if (stream_id >= num_streams - 1) return false;

    float boundary_y = (float(stream_id + 1)) * stream_step;
    return abs(uv.y - boundary_y) < line_width;
}

void main() {
    // Flip Y coordinate so top of screen = 0, bottom = 1
    float flipped_y = 1.0 - texCoord.y;

    // Calculate used height for all streams
    float used_height = stream_step * float(num_streams);

    // Check if outside used vertical space (streams are at top)
    if (flipped_y < 0.0 || flipped_y > used_height) {
        fragColor = vec4(0.0);
        return;
    }

    vec2 stream_uv = vec2(texCoord.x, flipped_y);
    vec2 frag_pos = vec2(texCoord.x, flipped_y);

    // Draw stream separator lines
    if (isBetweenStreamLine(stream_uv, stream_step, line_width / output_aspect_ratio * 0.5, num_streams, used_height)) {
        fragColor = vec4(0.5, 0.5, 0.5, 1.0);  // Grey separator lines
        return;
    }

    // Calculate stream ID and bounds
    int stream_id = int(floor(stream_uv.y / stream_step));
    float stream_center = (float(stream_id) + 0.5) * stream_step;
    float stream_top = stream_center - 0.5 * stream_step;

    // Sample from texture - flip v coordinate so first stream (index 0) is at top
    float texture_v = (float(num_streams - stream_id) - 0.5) / float(num_streams);
    vec2 value_uv = vec2(stream_uv.x, texture_v);
    float value = texture(tex0, value_uv).r;
    float mask = texture(tex0, value_uv).g;

    if (mask < 0.001) {
        fragColor = vec4(0.0);
        return;
    }

    float norm = clamp((value - display_range_min) / (display_range_max - display_range_min), 0.0, 1.0);
    // Invert norm so high values are at top, low values at bottom
    vec2 point = vec2(value_uv.x, stream_top + (1.0 - norm) * stream_step);

    // Draw continuous dots (no spacing)
    float dot_radius = line_width * 2.0;
    vec2 diff = frag_pos - point;
    // Correct for aspect ratio by dividing Y
    diff.y /= output_aspect_ratio;
    float dist = length(diff);
    if (dist < dot_radius) {
        int color_index = stream_id % num_colors;
        vec4 dot_color = colors[color_index];
        fragColor = vec4(dot_color.rgb, dot_color.a * alpha);
        return;
    }

    fragColor = vec4(0.0);
}
