#version 460 core

// Velocity Arrow Field Visualization
// Procedural arrows without geometry shaders. A 2D velocity is a white arrow; a 3D velocity (xyz) is
// a line for x/y with a dot at its end that grows with z, both white with no z, shading to red
// toward the camera and green away. With no x/y the dot stays on the grid point, so z still shows

uniform sampler2D tex0;
uniform bool xyz;                   // A 3D velocity: z in the blue channel
uniform float scale;
uniform float grid_spacing;         // in pixels
uniform float arrow_length;         // in pixels
uniform float arrow_thickness;
uniform float dot_min;              // smallest dot radius (px), with xyz
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

    // Arrow length scales with velocity, max at arrow_size
    float arrow_len = arrow_size * len;

    // Arrow shaft (from center to tip)
    vec2 start = vec2(0.0);
    vec2 end = dir * arrow_len;
    float shaft = line_segment(p, start, end, thickness);

    // Arrow head (two lines forming a V)
    float head_len = arrow_len * 0.35;
    float head_angle = 0.4;

    vec2 head_dir1 = vec2(
        dir.x * cos(head_angle) - dir.y * sin(head_angle),
        dir.x * sin(head_angle) + dir.y * cos(head_angle)
    );
    vec2 head_dir2 = vec2(
        dir.x * cos(-head_angle) - dir.y * sin(-head_angle),
        dir.x * sin(-head_angle) + dir.y * cos(-head_angle)
    );

    float head1 = line_segment(p, end, end - head_dir1 * head_len, thickness);
    float head2 = line_segment(p, end, end - head_dir2 * head_len, thickness);

    // Combine all parts
    return min(shaft, min(head1, head2));
}

// Distance to a line for x/y with a dot at its end, its radius growing with z
float line_dot(vec2 p, vec3 velocity, float arrow_size, float thickness) {
    vec2 end = velocity.xy * arrow_size;
    float line = length(end) < 0.001 ? 1e10 : line_segment(p, vec2(0.0), end, thickness);
    float radius = dot_min + 0.5 * arrow_size * abs(velocity.z);
    float dot_distance = radius > 0.0 ? length(p - end) - radius : 1e10;
    return min(line, dot_distance);
}

void main() {
    // Current pixel in screen space
    vec2 pixel = gl_FragCoord.xy;

    float min_dist = 1e10;
    float arrow_depth = 0.0;    // z of the nearest arrow

    // Check arrows from nearby grid cells (to allow arrows longer than grid spacing)
    // Max reach = arrow_length (velocity magnitude <= 1.0), plus the dot with xyz
    float reach = xyz ? 1.5 * arrow_length + dot_min : arrow_length;
    int search_radius = min(int(ceil(reach / grid_spacing)) + 1, 8);
    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            // Grid cell center
            vec2 grid_pos = floor(pixel / grid_spacing) * grid_spacing + grid_spacing * 0.5;
            grid_pos += vec2(dx, dy) * grid_spacing;

            // Sample velocity at this grid center
            vec2 grid_uv = grid_pos / resolution;
            vec3 velocity = texture(tex0, grid_uv).xyz * scale;

            // Position relative to this grid center
            vec2 local_pos = pixel - grid_pos;

            float dist = xyz ? line_dot(local_pos, velocity, arrow_length, arrow_thickness)
                             : arrow(local_pos, velocity.xy, arrow_length, arrow_thickness);
            if (dist < min_dist) {
                min_dist = dist;
                arrow_depth = xyz ? velocity.z : 0.0;
            }
        }
    }

    // High-quality anti-aliasing using derivative-based smoothing
    float edge_width = fwidth(min_dist) * 0.5;
    float alpha = 1.0 - smoothstep(-edge_width, edge_width, min_dist);

    vec3 color = vec3(1.0, 1.0, 1.0);
    if (arrow_depth != 0.0) {
        vec3 depth_color = arrow_depth > 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
        color = mix(color, depth_color, min(abs(arrow_depth), 1.0));
    }

    // Premultiply alpha to avoid dark edges
    fragColor = vec4(color * alpha, alpha);
}
