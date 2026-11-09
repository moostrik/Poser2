#version 460 core

// Channel View Shader
// A texture's channels as an image or as a colormap centered on mid grey, all of them or one; a channel the
// texture does not have reads as sampled (a one-channel texture's grey swizzle repeats x, w is 1)

uniform sampler2D tex0;
uniform float scale;     // Multiplier on the values
uniform int channel;     // 0 to 3 for x, y, z, w; 4 for all
uniform bool colormap;     // Centered on mid grey: x, y, z as red, green, blue, w as grey
uniform bool is_vector;    // With colormap, a vector: the color its direction's, its brightness its size, black at 0

in vec2 texCoord;
out vec4 fragColor;

const int ALL = 4;
const vec3 AXES[4] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(1.0));

void main() {
    vec4 value = texture(tex0, texCoord) * scale;

    vec3 color;
    if (colormap && is_vector) {
        vec3 v = channel == ALL ? value.xyz : AXES[channel] * value[channel];
        float size = length(v);
        color = size > 0.0 ? (0.5 + 0.5 * v / size) * min(size, 1.0) : vec3(0.0);
    } else if (colormap) {
        vec3 centered = channel == ALL ? value.xyz : AXES[channel] * value[channel];
        color = 0.5 + 0.5 * centered;
    } else if (channel == ALL) {
        color = value.xyz;
    } else {
        color = vec3(value[channel]);
    }
    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
