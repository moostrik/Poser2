#version 460 core

uniform int numJoints;
uniform int numColors;
uniform vec2 displayRange;
uniform float lineWidth = 0.01;
uniform float lineSmooth = 0.01;
uniform int useScores = 0;
uniform int useDeltas = 0;

uniform float values[32];
uniform float scores[32];
uniform float deltas[32];
uniform vec4 colors[8];  // Cycle through these colors

in vec2 texCoord;
out vec4 fragColor;

void main() {
    float joint_width = 1.0 / float(numJoints);
    int joint_index = int(texCoord.x / joint_width);

    if (joint_index >= numJoints) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float value = values[joint_index];
    float score = useScores != 0 ? scores[joint_index] : 1.0;

    if (useScores != 0 && score <= 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Get color by cycling through color array
    int color_index = joint_index % numColors;
    vec4 line_color = colors[color_index];

    // Normalize value to [0, 1]
    float normalized_value = (value - displayRange.x) / (displayRange.y - displayRange.x);

    float dist;
    float thickness = lineWidth;

    if (useDeltas != 0) {
        // Velocity-modulated mode: wrap value and modulate line thickness by delta
        normalized_value = fract(normalized_value);
        float delta = deltas[joint_index];
        thickness = max(abs(delta) * lineWidth * 10.0, lineWidth);
        dist = abs(texCoord.y - normalized_value);
        // Also check wrapped distance (for values that wrap around)
        dist = min(dist, min(abs(texCoord.y - (normalized_value + 1.0)),
                             abs(texCoord.y - (normalized_value - 1.0))));
        score = 1.0;  // Hardcode score in delta mode
    } else {
        // Standard mode: clamp value
        normalized_value = clamp(normalized_value, 0.0, 1.0);
        dist = abs(texCoord.y - normalized_value);
    }

    // Smooth antialiased line
    float alpha = 1.0 - smoothstep(thickness, thickness + lineSmooth, dist);

    if (alpha > 0.01) {
        fragColor = vec4(line_color.rgb, line_color.a * alpha * score);
    } else {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
}
