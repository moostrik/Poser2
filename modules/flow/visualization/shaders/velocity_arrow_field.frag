#version 460 core

// Velocity Arrow Field Visualization
// Procedural arrows without geometry shaders. A 2D velocity is a white arrow. A 3D velocity (xyz) is
// the same white arrow for x/y with a ring around the grid point for z, red toward the camera and
// green away, and shows with no x/y. Sizes are on the cell (spacing * element_length): at a speed
// of 1 the arrow's tip reaches 1 cell and the ring's radius is 1/pi cell, so a ring draws as much
// line as an arrow of the same value; faster, both roll off toward 2 cells

uniform sampler2D tex0;
uniform bool xyz;                   // A 3D velocity: z in the blue channel
uniform float scale;
uniform float z_scale;              // Multiplier on z alone
uniform float grid_spacing;         // in pixels
uniform float element_length;       // the cell's size, a ratio of grid_spacing
uniform float arrow_thickness;
uniform vec2 resolution;

in vec2 texCoord;
out vec4 fragColor;

#define PI 3.14159265
#define SIZE_LIMIT 2.0      // the speed the size rolls off toward

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

    // Arrow length scales with velocity
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

// Distance to a ring around the grid point, its radius |z| / pi on the arrow's scale
float ring(vec2 p, float z, float arrow_size, float thickness) {
    float radius = arrow_size * abs(z) / PI;
    if (radius < 0.001) return 1e10; // No ring for zero z
    return abs(length(p) - radius) - thickness;
}

// Anti-aliased coverage of a shape from its distance, by derivative-based smoothing
float coverage(float dist) {
    float edge_width = max(fwidth(dist) * 0.5, 1e-4);
    return 1.0 - smoothstep(-edge_width, edge_width, dist);
}

void main() {
    // Current pixel in screen space
    vec2 pixel = gl_FragCoord.xy;
    float cell = grid_spacing * element_length;

    float arrow_dist = 1e10;
    float ring_dist = 1e10;
    float ring_depth = 0.0;     // z of the nearest ring

    // Check arrows from nearby grid cells, out to the longest an arrow grows
    float reach = SIZE_LIMIT * cell + arrow_thickness;
    int search_radius = min(int(ceil(reach / grid_spacing)) + 1, 8);
    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            // Grid cell center
            vec2 grid_pos = floor(pixel / grid_spacing) * grid_spacing + grid_spacing * 0.5;
            grid_pos += vec2(dx, dy) * grid_spacing;

            // Sample velocity at this grid center, its size rolled off toward SIZE_LIMIT
            vec2 grid_uv = grid_pos / resolution;
            vec3 velocity = texture(tex0, grid_uv).xyz * scale;
            velocity.z = xyz ? velocity.z * z_scale : 0.0;
            float speed = length(velocity);
            if (speed > 0.001) velocity *= SIZE_LIMIT * tanh(speed / SIZE_LIMIT) / speed;

            // Position relative to this grid center
            vec2 local_pos = pixel - grid_pos;

            arrow_dist = min(arrow_dist, arrow(local_pos, velocity.xy, cell, arrow_thickness));
            if (xyz) {
                float dist = ring(local_pos, velocity.z, cell, arrow_thickness);
                if (dist < ring_dist) {
                    ring_dist = dist;
                    ring_depth = velocity.z;
                }
            }
        }
    }

    float arrow_alpha = coverage(arrow_dist);
    float ring_alpha = coverage(ring_dist);
    vec3 ring_color = ring_depth > 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);

    // The ring in its color, the white arrow over it; premultiplied alpha to avoid dark edges
    vec3 color = ring_color * ring_alpha * (1.0 - arrow_alpha) + vec3(arrow_alpha);
    float alpha = arrow_alpha + ring_alpha * (1.0 - arrow_alpha);
    fragColor = vec4(color, alpha);
}
