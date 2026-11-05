#version 460 core

// Log Grid Shader
// The lines of a log axis, 2 px grey at half opacity on transparent, written without blending: in each lane a line at every height given,
// and optionally a line down the middle, the histogram's 0

uniform int lanes;          // Lanes, top to bottom
uniform vec4 fractions;     // The lines' heights in a lane, 0 bottom, 1 top
uniform int count;          // The heights used, up to 4
uniform bool center;        // A line down the middle

in vec2 texCoord;
out vec4 fragColor;

const vec4 LINE = vec4(0.5, 0.5, 0.5, 0.5);

void main() {
    vec2 pixel = fwidth(texCoord);
    float height = fract(texCoord.y * float(lanes));
    float lane_pixel = pixel.y * float(lanes);
    bool line = center && abs(texCoord.x - 0.5) < pixel.x;             // 2 px wide
    for (int i = 0; i < count; i++) {
        line = line || abs(height - fractions[i]) < lane_pixel;
    }
    fragColor = line ? LINE : vec4(0.0);
}
