#version 460 core

// Time Series Shader
// One lane per series, newest on the right: each column holds the lowest and highest sample of its
// slice of time, normalized to the lane, so a band covers what happened and a spike stays visible

layout(binding = 0) uniform sampler2D   ranges;          // RG: lowest, highest per column (x) and lane (y); negative = no sample
uniform int         numLanes;
uniform int         numColumns;
uniform float       lineWidth;     // in lane heights
uniform vec4        colors[8];

in vec2     texCoord;
out vec4    fragColor;

vec2 column_range(int column, float texture_v) {
    return texture(ranges, vec2((float(column) + 0.5) / float(numColumns), texture_v)).rg;
}

void main() {
    // Top of the screen is 0, as in the window shader
    float flipped_y = 1.0 - texCoord.y;
    float lanes = flipped_y * float(numLanes);
    int lane = clamp(int(floor(lanes)), 0, numLanes - 1);
    float height = 1.0 - fract(lanes);                 // 0 at the bottom of the lane, 1 at its top
    vec4 color = colors[lane % 8];

    // The first lane is the texture's top row, as the window shader reads it
    float texture_v = (float(numLanes - lane) - 0.5) / float(numLanes);
    int column = clamp(int(texCoord.x * float(numColumns)), 0, numColumns - 1);
    vec2 range = column_range(column, texture_v);

    // Join to the column on the left, so a line has no gaps between columns
    if (range.x >= 0.0 && column > 0) {
        vec2 left = column_range(column - 1, texture_v);
        if (left.x >= 0.0) range = vec2(min(range.x, left.y), max(range.y, left.x));
    }

    float lane_edge = fract(lanes);
    if (lane_edge < lineWidth * 0.5) {
        fragColor = vec4(0.35, 0.35, 0.35, 1.0);        // separator between lanes
        return;
    }
    if (range.x >= 0.0 && height >= range.x - lineWidth && height <= range.y + lineWidth) {
        fragColor = color;
        return;
    }
    if (range.x >= 0.0 && height < range.x) {
        fragColor = vec4(color.rgb * 0.18, 1.0);        // faint fill below, to read the level at a glance
        return;
    }
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
