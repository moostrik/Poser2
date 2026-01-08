#version 460 core

// Velocity Arrow Field Visualization
// Procedural arrow rendering without geometry shaders

uniform sampler2D tex0;
uniform float scale;
uniform float grid_spacing;
uniform float arrow_scale;
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

    // Scale arrow by velocity magnitude
    float arrow_len = arrow_size * min(len, 1.0);

    // Arrow shaft (from center to tip)
    vec2 start = vec2(0.0);
    vec2 end = dir * arrow_len;
    float shaft = line_segment(p, start, end, thickness);

    // Arrow head (two lines forming a V)
    float head_len = arrow_len * 0.3;
    float head_angle = 0.4; // radians

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

    // Find nearest grid cell center
    vec2 grid_pos = floor(pixel / grid_spacing) * grid_spacing + grid_spacing * 0.5;

    // Sample velocity at grid center
    vec2 grid_uv = grid_pos / resolution;
    vec2 velocity = texture(tex0, grid_uv).xy * scale;

    // Position relative to grid center
    vec2 local_pos = pixel - grid_pos;

    // Calculate arrow distance (in pixel space)
    float arrow_size = grid_spacing * arrow_scale * 0.5;
    float thickness = grid_spacing * arrow_thickness * 0.05;
    float dist = arrow(local_pos, velocity * grid_spacing * 0.4, arrow_size, thickness);

    // Anti-aliased rendering
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);

    // Color based on velocity magnitude
    float mag = length(velocity);
    vec3 color = mix(vec3(0.3, 0.5, 0.7), vec3(1.0, 0.2, 0.3), clamp(mag, 0.0, 1.0));

    fragColor = vec4(color, alpha);
}
