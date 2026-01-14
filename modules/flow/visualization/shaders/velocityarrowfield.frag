#version 460 core

// layout(origin_upper_left) in vec4 gl_FragCoord;

// Velocity Arrow Field Visualization
// Procedural arrow rendering without geometry shaders

uniform sampler2D tex0;
uniform float scale;
uniform float grid_spacing;
uniform float arrow_length;
uniform float arrow_thickness;
uniform vec2 resolution;

in vec2 texCoord;
out vec4 fragColor;

// Distance to line segment
float line_segment(vec2 p, vec2 a, vec2 b, float thickness) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - thickness;
}

// Distance to arrow shape (shaft + head)
float arrow(vec2 p, vec2 velocity, float arrow_size, float thickness) {
    float len = length(velocity);
    if (len < 0.001) return 1e10; // No arrow for zero velocity

    vec2 dir = velocity / len;
    vec2 perp = vec2(-dir.y, dir.x);

    // Arrow length scaled by velocity magnitude (clamped to arrow_size max)
    float arrow_len = arrow_size * min(len, 1.0);

    // Arrow shaft (from center to tip)
    vec2 start = vec2(0.0);
    vec2 end = dir * arrow_len;
    float shaft = line_segment(p, start, end, thickness);

    // Arrow head (two lines forming a V)
    float head_len = arrow_len * 0.25;
    float head_angle = 0.1; // radians

    vec2 head_dir1 = vec2(
        dir.x * cos(head_angle) - dir.y * sin(head_angle),
        dir.x * sin(head_angle) + dir.y * cos(head_angle)
    );
    vec2 head_dir2 = vec2(
        dir.x * cos(-head_angle) - dir.y * sin(-head_angle),
        dir.x * sin(-head_angle) + dir.y * cos(-head_angle)
    );

    vec2 head_start1 = end;
    vec2 head_end1 = end - head_dir1 * head_len;
    float head1 = line_segment(p, head_start1, head_end1, thickness);

    vec2 head_start2 = end;
    vec2 head_end2 = end - head_dir2 * head_len;
    float head2 = line_segment(p, head_start2, head_end2, thickness);

    // Combine all parts
    return min(shaft, min(head1, head2));
}

void main() {
    // Current pixel in screen space
    vec2 pixel = gl_FragCoord.xy;

    float min_dist = 1e10;

    // Check arrows from nearby grid cells (to allow arrows longer than grid spacing)
    int search_radius = int(ceil(arrow_length / grid_spacing)) + 1;
    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            // Grid cell center
            vec2 grid_pos = floor(pixel / grid_spacing) * grid_spacing + grid_spacing * 0.5;
            grid_pos += vec2(dx, dy) * grid_spacing;

            // Sample velocity at this grid center
            vec2 grid_uv = grid_pos / resolution;
            vec2 velocity = texture(tex0, grid_uv).xy * scale;

            // Position relative to this grid center
            vec2 local_pos = pixel - grid_pos;

            // Calculate arrow distance
            float dist = arrow(local_pos, velocity, arrow_length, arrow_thickness);
            min_dist = min(min_dist, dist);
        }
    }

    // High-quality anti-aliasing using derivative-based smoothing
    float edge_width = fwidth(min_dist) * 0.5;
    float alpha = 1.0 - smoothstep(-edge_width, edge_width, min_dist);

    // Simple color (matching original shader style)
    vec3 color = vec3(1.0, 1.0, 1.0);

    // Premultiply alpha to avoid dark edges
    fragColor = vec4(color * alpha, alpha);
}
