#version 460 core

// Velocity Arrow Field Visualization
// Procedural markers without geometry shaders, one outlined blob per grid point: a round body for z
// and a bulb on a stem reaching out along x/y. Pure z is a circle, pure x/y a stem with a bulb at
// its tip. Sizes are on the cell (spacing * element_length): at a speed of 1 the body's radius is
// 0.4 cell and the bulb's tip reaches 1 cell; faster, the blob rolls off toward 2 cells. The body
// is red toward the camera and green away, the stem and bulb white

uniform sampler2D tex0;
uniform bool xyz;                   // A 3D velocity: z in the blue channel
uniform float scale;
uniform float grid_spacing;         // in pixels
uniform float element_length;       // the cell's size, a ratio of grid_spacing
uniform float arrow_thickness;      // outline, in pixels
uniform vec2 resolution;

in vec2 texCoord;
out vec4 fragColor;

#define BODY_RADIUS 0.4     // cells, at a speed of 1
#define REACH 1.0           // cells from the grid point to the bulb's tip, at a speed of 1
#define BULB_RADIUS 0.15    // of the reach
#define NECK_RADIUS 0.5     // of the bulb's radius
#define SIZE_LIMIT 2.0      // the speed the size rolls off toward
#define FAR 1e10

// Distance to a line segment
float segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// Polynomial smooth minimum: the neck between the parts
float smooth_min(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

// Signed distance to one blob; `body` is the distance to its body alone, for the color
float blob(vec2 p, vec3 velocity, float cell, out float body) {
    float speed = length(velocity);
    body = FAR;
    if (speed < 0.001) return FAR;
    velocity *= SIZE_LIMIT * tanh(speed / SIZE_LIMIT) / speed;

    float body_radius = BODY_RADIUS * cell * abs(velocity.z);
    if (body_radius > 0.001) body = length(p) - body_radius;

    float reach = REACH * cell * length(velocity.xy);
    if (reach < 0.001) return body;

    vec2 dir = normalize(velocity.xy);
    float bulb_radius = BULB_RADIUS * reach;
    vec2 bulb_center = dir * (reach - bulb_radius);
    float bulb = length(p - bulb_center) - bulb_radius;
    float stem = segment(p, vec2(0.0), bulb_center) - NECK_RADIUS * bulb_radius;
    float tail = smooth_min(stem, bulb, bulb_radius);
    return body < FAR ? smooth_min(body, tail, bulb_radius) : tail;
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

    float outline_dist = FAR;
    float body_share = 0.0;     // how much of the nearest outline is body, for its color
    float depth = 0.0;          // z of the nearest blob

    // Check blobs from nearby grid cells, out to the largest a blob grows
    float reach = SIZE_LIMIT * REACH * cell + arrow_thickness;
    int search_radius = min(int(ceil(reach / grid_spacing)) + 1, 8);
    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            // Grid cell center
            vec2 grid_pos = floor(pixel / grid_spacing) * grid_spacing + grid_spacing * 0.5;
            grid_pos += vec2(dx, dy) * grid_spacing;

            // Sample velocity at this grid center
            vec2 grid_uv = grid_pos / resolution;
            vec3 velocity = texture(tex0, grid_uv).xyz * scale;
            if (!xyz) velocity.z = 0.0;

            float body;
            float dist = abs(blob(pixel - grid_pos, velocity, cell, body)) - arrow_thickness;
            if (dist < outline_dist) {
                outline_dist = dist;
                body_share = clamp(1.0 - (abs(body) - arrow_thickness - dist) / max(arrow_thickness * 4.0, 1.0), 0.0, 1.0);
                depth = velocity.z;
            }
        }
    }

    float alpha = coverage(outline_dist);
    vec3 body_color = depth > 0.0 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 color = mix(vec3(1.0), body_color, body_share);

    // Premultiplied alpha to avoid dark edges
    fragColor = vec4(color * alpha, alpha);
}
