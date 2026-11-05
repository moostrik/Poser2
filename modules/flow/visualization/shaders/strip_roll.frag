#version 460 core

// Strip Roll Shader
// A rolling plot, drawn on the GPU so nothing is read back: everything moves one column left
// and the last column gets one bar per lane, the frame-wide mean amount of that channel

uniform sampler2D tex0;  // Previous strip
uniform sampler2D tex1;  // Summary, its top mip the mean amount of (x, y, z, w)
uniform float scale;     // Multiplier on the bars

out vec4 fragColor;

const int LANES = 4;
const vec3 COLORS[LANES] = vec3[](vec3(1.0, 0.35, 0.3), vec3(0.3, 1.0, 0.4), vec3(0.3, 0.55, 1.0), vec3(1.0));

void main() {
    ivec2 size = textureSize(tex0, 0);
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (pixel.x < size.x - 1) {
        fragColor = texelFetch(tex0, pixel + ivec2(1, 0), 0);
        return;
    }

    vec4 means = textureLod(tex1, vec2(0.5), float(textureQueryLevels(tex1) - 1));
    float lanes = float(pixel.y) * float(LANES) / float(size.y);
    int lane = LANES - 1 - int(lanes);                      // the first lane on top
    float height = fract(lanes);
    bool bar = height < means[lane] * scale && height < 0.94;     // a gap between the lanes
    fragColor = vec4(bar ? COLORS[lane] : vec3(0.0), 1.0);
}
