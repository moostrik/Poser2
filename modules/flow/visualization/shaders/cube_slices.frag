#version 460 core

// Cube Slices: each layer's texels drawn as the channel view draws them (as stored, or as a colormap
// centered on mid grey), partly transparent so the layers show through each other; the selected
// layer opaque and at full brightness

uniform sampler3D tex0;  // The volume
uniform float scale;     // Multiplier on the values
uniform int channel;     // 0 to 3 for x, y, z, w; 4 for all
uniform bool colormap;   // Centered on mid grey: x, y, z as red, green, blue, w as grey
uniform float opacity;   // The other layers' opacity

in vec2 texCoord;
in float w;
flat in int is_selected;
out vec4 fragColor;

const int ALL = 4;
const vec3 AXES[4] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(1.0));

void main() {
    vec4 value = texture(tex0, vec3(texCoord, w)) * scale;
    vec3 color;
    if (colormap) {
        vec3 centered = channel == ALL ? value.xyz : AXES[channel] * value[channel];
        color = 0.5 + 0.5 * centered;
    } else if (channel == ALL) {
        color = value.xyz;
    } else {
        color = vec3(value[channel]);
    }
    bool chosen = is_selected == 1;
    fragColor = vec4(clamp(color, 0.0, 1.0) * (chosen ? 1.0 : 0.75), chosen ? 1.0 : opacity);
}
