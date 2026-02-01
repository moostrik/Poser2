#version 460 core

uniform sampler2D   tex0;
uniform float       sample_step;
uniform int         num_streams;
uniform float       stream_step;
uniform float       line_width;
uniform float       output_aspect_ratio;  // output buffer aspect ratio (width/height)
uniform float       display_range;  // max absolute value for display (e.g., pi)

in vec2     texCoord;
in vec2     clipPos;
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

    float norm = abs(value) / display_range;
    vec2 point = vec2(value_uv.x, stream_top + norm * stream_step);

    // Draw dots at intervals
    float dot_spacing = sample_step;  // Adjust multiplier to control spacing
    float nearest_dot_x = (floor(texCoord.x / dot_spacing) + 0.5) * dot_spacing;

    // Sample value at the nearest dot position
    vec2 dot_value_uv = vec2(nearest_dot_x, stream_center);
    float dot_value = texture(tex0, dot_value_uv).r;
    float dot_mask = texture(tex0, dot_value_uv).g;

    if (dot_mask > 0.001) {
        float dot_norm = abs(dot_value) / display_range;
        vec2 dot_point = vec2(nearest_dot_x, stream_top + dot_norm * stream_step);

        // Draw a dot at the point position (accounting for aspect ratio)
        float dot_radius = sample_step * 0.5;  // Adjust size as needed
        vec2 diff = frag_pos - dot_point;
        diff.x *= output_aspect_ratio;  // Account for aspect ratio
        float dist = length(diff);
        if (dist < dot_radius) {
            // Color the dot based on value sign (red for positive, blue for negative)
            vec3 dot_color = dot_value >= 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 0.0, 1.0);
            fragColor = vec4(dot_color, 1.0);
            return;
        }
    }

    fragColor = vec4(0.0);
}
