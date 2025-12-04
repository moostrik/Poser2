#version 460 core

uniform int num_points;
uniform int num_segments;
uniform float line_width = 0.01;
uniform float line_smooth = 0.01;
uniform float aspect_ratio = 1.0;  // width / height

// Array of packed point data: [x, y, score, visibility]
uniform vec4 points[64];  // Max 64 points

// Array of line segment indices [start_idx, end_idx]
uniform ivec2 segments[32];  // Max 32 line segments

in vec2 texCoord;
out vec4 fragColor;

// Calculate distance from point to line segment
float distanceToSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    float totalAlpha = 0.0;

    // Apply aspect ratio correction to texture coordinates
    vec2 correctedTexCoord = texCoord;
    correctedTexCoord.x *= aspect_ratio;

    // Iterate through all line segments
    for (int i = 0; i < num_segments; i++) {
        // Get line segment indices
        ivec2 segment = segments[i];
        int idx_a = segment.x;
        int idx_b = segment.y;

        // Fetch endpoint data
        vec4 point_a = points[idx_a];
        vec4 point_b = points[idx_b];

        vec2 pos_a = point_a.xy;
        vec2 pos_b = point_b.xy;
        float score_a = point_a.z;
        float score_b = point_b.z;
        float vis_a = point_a.w;
        float vis_b = point_b.w;

        // Skip invalid segments
        if (vis_a < 0.5 || vis_b < 0.5 || pos_a.x < 0.0 || pos_b.x < 0.0) continue;

        // Apply aspect ratio correction to positions
        vec2 corrected_a = pos_a;
        vec2 corrected_b = pos_b;
        corrected_a.x *= aspect_ratio;
        corrected_b.x *= aspect_ratio;

        // Calculate distance from fragment to line segment
        float dist = distanceToSegment(correctedTexCoord, corrected_a, corrected_b);

        // Create smooth line
        float alpha = 1.0 - smoothstep(line_width - line_smooth, line_width + line_smooth, dist);

        // Apply average score-based alpha
        float avg_score = (score_a + score_b) * 0.5;
        alpha *= avg_score;

        // Accumulate alpha with proper blending: new_alpha = old_alpha + new * (1 - old_alpha)
        totalAlpha = totalAlpha + alpha * (1.0 - totalAlpha);
    }

    // Output white with the accumulated alpha
    fragColor = vec4(1.0, 1.0, 1.0, totalAlpha);
}