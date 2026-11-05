#version 460 core

// Colormap Key Shader
// The colormap's key: in a box, one bar per channel shown, each running from -1 at its
// left through 0 (mid grey) to 1 at its right, in the color the colormap gives that channel

uniform vec4 box;           // x0, y0, x1, y1 in pixels from the bottom left
uniform int channel;        // 0 to 3 for one bar, 4 for x, y, z
uniform float bar_height;   // in pixels
uniform float bar_gap;      // in pixels, between the bars

out vec4 fragColor;

const int ALL = 4;
const vec3 AXES[4] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(1.0));

void main() {
    vec2 p = gl_FragCoord.xy;
    fragColor = vec4(0.0);
    if (p.x < box.x || p.x > box.z || p.y < box.y || p.y > box.w) return;

    int bars = channel == ALL ? 3 : 1;
    float from_top = box.w - p.y;
    int bar = int(from_top / (bar_height + bar_gap));
    if (bar >= bars || from_top - float(bar) * (bar_height + bar_gap) > bar_height) return;

    int bar_channel = channel == ALL ? bar : channel;
    float t = mix(-1.0, 1.0, (p.x - box.x) / (box.z - box.x));
    fragColor = vec4(clamp(0.5 + 0.5 * AXES[bar_channel] * t, 0.0, 1.0), 1.0);
}
