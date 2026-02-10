#version 460 core

uniform float line_width = 0.01;
uniform float line_smooth = 0.01;
uniform float aspect_ratio = 1.0;
uniform vec4 line_color = vec4(-1.0, -1.0, -1.0, -1.0);
uniform vec4 points[17];

const int NUM_SEGMENTS = 16;
const ivec2 segments[NUM_SEGMENTS] = ivec2[NUM_SEGMENTS](
    ivec2(0, 1), ivec2(0, 2), ivec2(1, 3), ivec2(2, 4),
    ivec2(5, 6), ivec2(5, 7), ivec2(7, 9), ivec2(6, 8), ivec2(8, 10),
    ivec2(5, 11), ivec2(6, 12), ivec2(11, 12),
    ivec2(11, 13), ivec2(13, 15), ivec2(12, 14), ivec2(14, 16)
);

// Anatomical colors - synchronized with modules/render/layers/colors.py
const vec3 POSE_COLOR_CENTER = vec3(1.0, 1.0, 1.0);    // White
const vec3 POSE_COLOR_LEFT = vec3(1.0, 0.5, 0.0);      // Orange
const vec3 POSE_COLOR_RIGHT = vec3(0.0, 1.0, 1.0);     // Cyan

const vec3 joint_colors[17] = vec3[17](
    POSE_COLOR_CENTER,
    POSE_COLOR_LEFT, POSE_COLOR_RIGHT, POSE_COLOR_LEFT, POSE_COLOR_RIGHT,
    POSE_COLOR_LEFT, POSE_COLOR_RIGHT, POSE_COLOR_LEFT, POSE_COLOR_RIGHT,
    POSE_COLOR_LEFT, POSE_COLOR_RIGHT, POSE_COLOR_LEFT, POSE_COLOR_RIGHT,
    POSE_COLOR_LEFT, POSE_COLOR_RIGHT, POSE_COLOR_LEFT, POSE_COLOR_RIGHT
);

in vec2 texCoord;
out vec4 fragColor;

float distanceToSegment(vec2 p, vec2 a, vec2 b, out float t) {
    vec2 ba = b - a;
    t = clamp(dot(p - a, ba) / dot(ba, ba), 0.0, 1.0);
    return length(p - a - ba * t);
}

void processSegment(int idx_a, int idx_b, vec2 pos, inout vec3 colorAccum, inout float alphaAccum) {
    vec4 pa = points[idx_a];
    vec4 pb = points[idx_b];

    if (pa.w < 0.5 || pb.w < 0.5) return;

    vec2 a = vec2(pa.x * aspect_ratio, pa.y);
    vec2 b = vec2(pb.x * aspect_ratio, pb.y);

    float t;
    float dist = distanceToSegment(pos, a, b, t);
    float alpha = 1.0 - smoothstep(line_width - line_smooth, line_width + line_smooth, dist);
    alpha *= mix(pa.z, pb.z, t) * 0.5 + 0.5;

    vec3 color = mix(joint_colors[idx_a], joint_colors[idx_b], t);
    if (line_color.a >= 0.0) {
        color = line_color.rgb;
        alpha *= line_color.a;
    }

    colorAccum += color * alpha;
    alphaAccum += alpha * (1.0 - alphaAccum);
}

void main() {
    vec2 pos = texCoord * vec2(aspect_ratio, 1.0);
    vec3 colorAccum = vec3(0.0);
    float alphaAccum = 0.0;

    for (int i = 0; i < NUM_SEGMENTS; i++) {
        processSegment(segments[i].x, segments[i].y, pos, colorAccum, alphaAccum);
    }

    vec3 color = alphaAccum > 0.001 ? colorAccum / alphaAccum : vec3(0.0);
    fragColor = vec4(color, alphaAccum);
}