#version 460 core

uniform int num_points;
uniform float dot_size = 0.01;
uniform float dot_smooth = 0.01;
uniform float aspect_ratio = 1.0;  // width / height

// Array of packed point data: [x, y, score, visibility]
uniform vec4 points[64];  // Max 64 points, more than enough for pose keypoints

in vec2 texCoord;
out vec4 fragColor;

void main() {
    vec4 color = vec4(0.0);

    // Iterate through all points
    for (int i = 0; i < num_points; i++) {
        // Fetch packed data: [x, y, score, visibility]
        vec4 point_data = points[i];
        vec2 pos = point_data.xy;
        float score = point_data.z;
        float visibility = point_data.w;

        // Skip invalid points
        if (visibility < 0.5 || pos.x < 0.0) continue;

        // Calculate distance with aspect ratio correction
        vec2 delta = texCoord - pos;
        delta.x *= aspect_ratio;  // Scale X by aspect ratio
        float dist = length(delta);

        // Create smooth circle
        float alpha = 1.0 - smoothstep(dot_size - dot_smooth, dot_size + dot_smooth, dist);

        // Apply score-based alpha
        alpha *= score;

        // Use premultiplied alpha blending
        vec3 dot_color = vec3(1.0) * alpha;  // White premultiplied by alpha
        color.rgb = color.rgb + dot_color * (1.0 - color.a);
        color.a = color.a + alpha * (1.0 - color.a);
    }

    fragColor = color;
}