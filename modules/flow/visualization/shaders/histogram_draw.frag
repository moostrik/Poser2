#version 460 core

// Histogram Draw Shader
// Draws the bins: x from -1 at the left through 0 in the middle to 1 at the right, the height the
// log of the count against the log of the samples. Four lanes, x on top, or one channel's lane at
// full height

uniform usampler2D tex0;    // The bins, one row per channel
uniform int channel;        // 0 to 3 for one channel, 4 for all four
uniform float total;        // Samples counted, the count at full height

in vec2 texCoord;
out vec4 fragColor;

const int ALL = 4;
const vec3 COLORS[4] = vec3[](vec3(1.0, 0.35, 0.3), vec3(0.3, 1.0, 0.4), vec3(0.3, 0.55, 1.0), vec3(1.0));

float height_at(int lane_channel, float x) {
    int bins = textureSize(tex0, 0).x;
    int bin = clamp(int(x * float(bins)), 0, bins - 1);
    uint count = texelFetch(tex0, ivec2(bin, lane_channel), 0).r;
    return log(1.0 + float(count)) / log(1.0 + total);
}

void main() {
    vec2 p = texCoord;
    vec3 color = vec3(0.0);

    int lanes = channel == ALL ? 4 : 1;
    float position = p.y * float(lanes);
    int lane_channel = channel == ALL ? lanes - 1 - int(position) : channel;   // the first lane on top
    float height = fract(position);
    bool bar = height < height_at(lane_channel, p.x) * 0.94;               // a gap between the lanes
    if (bar) color = COLORS[lane_channel];
    fragColor = vec4(color, 1.0);
}
