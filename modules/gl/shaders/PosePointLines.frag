#version 460 core

uniform float line_width = 0.01;
uniform float line_smooth = 0.01;
uniform float aspect_ratio = 1.0;  // width / height
uniform vec4 line_color = vec4(-1.0, -1.0, -1.0, -1.0);  // Negative alpha means "not set"

// Array of packed point data: [x, y, score, visibility]
uniform vec4 points[17];  // Exactly 17 body landmarks

// Skeleton connectivity as constant
const int NUM_SEGMENTS = 16;
const ivec2 segments[NUM_SEGMENTS] = ivec2[NUM_SEGMENTS](
    ivec2(0, 1), ivec2(0, 2), ivec2(1, 3), ivec2(2, 4),  // Face
    ivec2(5, 6), ivec2(5, 7), ivec2(7, 9), ivec2(6, 8), ivec2(8, 10),  // Arms
    ivec2(5, 11), ivec2(6, 12), ivec2(11, 12),  // Torso
    ivec2(11, 13), ivec2(13, 15), ivec2(12, 14), ivec2(14, 16)  // Legs
);

// Per-joint colors (matching POSE_JOINT_COLORS from PoseMeshUtils.py)
const vec3 POSE_COLOR_CENTER = vec3(1.0, 1.0, 1.0);  // White
const vec3 POSE_COLOR_LEFT = vec3(1.0, 0.5, 0.0);    // Orange
const vec3 POSE_COLOR_RIGHT = vec3(0.0, 1.0, 1.0);   // Cyan

const vec3 joint_colors[17] = vec3[17](
    POSE_COLOR_CENTER,  // 0: nose
    POSE_COLOR_LEFT,    // 1: left_eye
    POSE_COLOR_RIGHT,   // 2: right_eye
    POSE_COLOR_LEFT,    // 3: left_ear
    POSE_COLOR_RIGHT,   // 4: right_ear
    POSE_COLOR_LEFT,    // 5: left_shoulder
    POSE_COLOR_RIGHT,   // 6: right_shoulder
    POSE_COLOR_LEFT,    // 7: left_elbow
    POSE_COLOR_RIGHT,   // 8: right_elbow
    POSE_COLOR_LEFT,    // 9: left_wrist
    POSE_COLOR_RIGHT,   // 10: right_wrist
    POSE_COLOR_LEFT,    // 11: left_hip
    POSE_COLOR_RIGHT,   // 12: right_hip
    POSE_COLOR_LEFT,    // 13: left_knee
    POSE_COLOR_RIGHT,   // 14: right_knee
    POSE_COLOR_LEFT,    // 15: left_ankle
    POSE_COLOR_RIGHT    // 16: right_ankle
);

in vec2 texCoord;
out vec4 fragColor;

// Calculate distance from point to line segment and return both distance and interpolation factor
void distanceToSegment(vec2 p, vec2 a, vec2 b, out float dist, out float t) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    t = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    dist = length(pa - ba * t);
}

void main() {
    vec3 totalColor = vec3(0.0);
    float totalAlpha = 0.0;

    // Apply aspect ratio correction to texture coordinates
    vec2 correctedTexCoord = texCoord;
    correctedTexCoord.x *= aspect_ratio;

    // Check if custom color is set (uniform branch, but can be optimized out)
    bool useCustomColor = line_color.a >= 0.0;

    // Iterate through all line segments
    for (int i = 0; i < NUM_SEGMENTS; i++) {
        // Get line segment indices
        ivec2 segment = segments[i];
        int idx_a = segment.x;
        int idx_b = segment.y;

        // Fetch endpoint data
        vec4 point_a = points[idx_a];
        vec4 point_b = points[idx_b];

        vec2 pos_a = point_a.xy;
        vec2 pos_b = point_b.xy;
        float score_a = point_a.z * 0.5 + 0.5;
        float score_b = point_b.z * 0.5 + 0.5;
        float vis_a = point_a.w;
        float vis_b = point_b.w;

        // Skip invalid segments
        if (vis_a < 0.5 || vis_b < 0.5) continue;

        // Apply aspect ratio correction to positions
        vec2 corrected_a = pos_a;
        vec2 corrected_b = pos_b;
        corrected_a.x *= aspect_ratio;
        corrected_b.x *= aspect_ratio;

        // Calculate distance from fragment to line segment and get interpolation factor
        float dist, t;
        distanceToSegment(correctedTexCoord, corrected_a, corrected_b, dist, t);

        // Create smooth line
        float alpha = 1.0 - smoothstep(line_width - line_smooth, line_width + line_smooth, dist);

        // Apply interpolated score-based alpha (score varies along the segment)
        float interpolated_score = mix(score_a, score_b, t);
        alpha *= interpolated_score;

        // Interpolate between joint colors based on position along segment
        vec3 gradientColor = mix(joint_colors[idx_a], joint_colors[idx_b], t);

        // Choose between custom color and gradient color
        vec3 segmentColor = mix(gradientColor, line_color.rgb, float(useCustomColor));

        // Apply custom color alpha if set
        alpha *= mix(1.0, line_color.a, float(useCustomColor));

        // Accumulate color and alpha with proper blending
        totalColor = totalColor + segmentColor * alpha * (1.0 - totalAlpha);
        totalAlpha = totalAlpha + alpha * (1.0 - totalAlpha);
    }

    // Output accumulated color and alpha
    fragColor = vec4(totalColor, totalAlpha);
}