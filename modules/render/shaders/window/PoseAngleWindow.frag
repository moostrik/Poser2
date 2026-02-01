#version 460 core

uniform sampler2D   tex0;
uniform float       sample_step;
uniform int         num_streams;
uniform float       stream_step;
uniform float       line_width;
uniform float       output_aspect_ratio;  // output buffer aspect ratio (width/height)
uniform float       display_range;  // max absolute value for display (e.g., pi)

in vec2     texCoord;
out vec4    fragColor;

bool isBetweenStreamLine(vec2 uv, float stream_step, float line_width, int num_streams) {
    int stream_id = int(floor(uv.y / stream_step));
    if (stream_id >= num_streams - 1) return false;

    float boundary_y = (float(stream_id + 1)) * stream_step;
    return abs(uv.y - boundary_y) < line_width;
}

void main() {
    vec2 uv = vec2(texCoord.x, 1.0 - texCoord.y);
    vec2 frag_pos = vec2(texCoord.x, uv.y);

    // Draw stream separator lines
    if (isBetweenStreamLine(uv, stream_step, line_width / output_aspect_ratio * 0.5, num_streams)) {
        fragColor = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    // Calculate stream ID and bounds
    int stream_id = int(floor(uv.y / stream_step));
    float stream_center = (float(stream_id) + 0.5) * stream_step;
    float stream_top = stream_center - 0.5 * stream_step;

    // Check multiple sample pairs to catch steep lines
    float min_dist = 1000.0;
    float best_conf = 0.0;

    for (int i = -2; i <= 2; i++) {
        float check_x = texCoord.x + float(i) * sample_step;
        if (check_x < 0.0 || check_x > 1.0) continue;

        vec2 check_curr_uv = vec2(check_x, stream_center);
        vec2 check_prev_uv = vec2(max(0.0, check_x - sample_step), stream_center);

        vec2 check_curr_sample = texture(tex0, check_curr_uv).rg;
        vec2 check_prev_sample = texture(tex0, check_prev_uv).rg;

        // Map raw value to stream position: 0 at center, ±display_range at edges
        float curr_norm = clamp(check_curr_sample.r / display_range * 0.5 + 0.5, 0.0, 1.0);
        float prev_norm = clamp(check_prev_sample.r / display_range * 0.5 + 0.5, 0.0, 1.0);

        vec2 check_curr_point = vec2(check_curr_uv.x, stream_top + curr_norm * stream_step);
        vec2 check_prev_point = vec2(check_prev_uv.x, stream_top + prev_norm * stream_step);

        vec2 line_vec = check_curr_point - check_prev_point;
        float line_len = length(line_vec);

        if (line_len < 0.0001) continue;

        vec2 to_frag = frag_pos - check_prev_point;
        float t = clamp(dot(to_frag, line_vec) / (line_len * line_len), 0.0, 1.0);
        vec2 closest = check_prev_point + t * line_vec;

        vec2 dist_vec = frag_pos - closest;
        dist_vec.x *= output_aspect_ratio;
        float dist = length(dist_vec);

        if (dist < min_dist) {
            min_dist = dist;
            best_conf = mix(check_prev_sample.g, check_curr_sample.g, t);
        }
    }

    // Antialiasing: smooth falloff at line edges
    float aa_width = line_width * 0.5;
    float line_alpha = 1.0 - smoothstep(line_width - aa_width, line_width + aa_width, min_dist);

    // Draw if within antialiased line width
    if (line_alpha > 0.001) {
        // Green for angle visualization
        vec3 color = vec3(0.2, 1.0, 0.3);
        fragColor = vec4(color * best_conf * line_alpha, line_alpha);
        return;
    }

    fragColor = vec4(0.0);
}
