#version 460 core

// Strip Roll Shader
// A rolling plot, drawn on the GPU so nothing is read back: everything moves one column left
// and the last column gets one bar per lane, the frame-wide mean amount of that channel on a log
// axis of three decades; a lane for each channel the texture holds, or one channel's lane at full height

uniform sampler2D tex0;  // Previous strip
uniform sampler2D tex1;  // Summary, its top mip the mean amount of (x, y, z, w)
uniform float scale;     // Multiplier on the bars
uniform int channel;     // 0 to 3 for one lane, 4 for all the texture holds
uniform int channels;    // The channels the texture holds, x first, 1 to 4

out vec4 fragColor;

const int ALL = 4;
const float DECADES = 3.0;       // the lane's height: a mean times scale from 0.001 at the bottom to 1 at the top
const vec3 COLORS[4] = vec3[](vec3(1.0, 0.35, 0.3), vec3(0.3, 1.0, 0.4), vec3(0.3, 0.55, 1.0), vec3(1.0));

void main() {
    ivec2 size = textureSize(tex0, 0);
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (pixel.x < size.x - 1) {
        fragColor = texelFetch(tex0, pixel + ivec2(1, 0), 0);
        return;
    }

    vec4 means = textureLod(tex1, vec2(0.5), float(textureQueryLevels(tex1) - 1));
    int lanes = channel == ALL ? channels : 1;
    float position = float(pixel.y) * float(lanes) / float(size.y);
    int lane = channel == ALL ? lanes - 1 - int(position) : channel;     // the first lane on top
    float height = fract(position);
    float level = clamp((log(max(means[lane] * scale, 1e-6)) / log(10.0) + DECADES) / DECADES, 0.0, 1.0);
    bool bar = height < level * 0.94;                                   // a gap between the lanes
    fragColor = vec4(bar ? COLORS[lane] : vec3(0.0), 1.0);
}
