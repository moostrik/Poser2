#version 460 core

// Volume Fog: a volume drawn continuously in its box, one ray per pixel from the orbit camera,
// marched front to back. Each sample is colored as the channel view colors it (as stored, or as a
// colormap centered on grey) and its amount is its opacity: an image's largest channel, a scalar's
// or a velocity's size. Out comes the premultiplied color and its coverage

uniform sampler3D tex0;               // The volume
uniform mat4 inverse_view_projection; // clip space back to the box's space
uniform vec3 box;                     // the box's size, its width 1
uniform float scale;                  // Multiplier on the values
uniform int channel;                  // 0 to 3 for x, y, z, w; 4 for all
uniform bool colormap;                // Centered on grey, instead of as stored
uniform int steps;                    // samples along a ray through the whole box
uniform float density;                // opacity per box width at an amount of 1

in vec2 texCoord;
out vec4 fragColor;

const int ALL = 4;
const vec3 AXES[4] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(1.0));

vec3 unproject(vec2 ndc, float depth) {
    vec4 p = inverse_view_projection * vec4(ndc, depth, 1.0);
    return p.xyz / p.w;
}

void main() {
    vec2 ndc = texCoord * 2.0 - 1.0;
    vec3 origin = unproject(ndc, -1.0);
    vec3 dir = normalize(unproject(ndc, 1.0) - origin);

    // Where the ray is inside the box (the slab test)
    vec3 half_box = 0.5 * box;
    vec3 inv = 1.0 / dir;
    vec3 t0 = (-half_box - origin) * inv;
    vec3 t1 = (half_box - origin) * inv;
    vec3 near = min(t0, t1);
    vec3 far = max(t0, t1);
    float enter = max(max(near.x, near.y), max(near.z, 0.0));
    float leave = min(min(far.x, far.y), far.z);
    if (leave <= enter) {
        fragColor = vec4(0.0);
        return;
    }

    float step_length = length(box) / float(steps);
    vec3 color = vec3(0.0);
    float coverage = 0.0;
    for (float t = enter + 0.5 * step_length; t < leave && coverage < 0.99; t += step_length) {
        vec3 p = origin + dir * t;
        vec3 uvw = vec3(p.x / box.x + 0.5, p.y / box.y + 0.5, 0.5 - p.z / box.z);   // the front, layer 0, toward +z
        vec4 value = texture(tex0, uvw) * scale;
        vec3 sample_color;
        float amount;
        if (colormap) {
            vec3 centered = channel == ALL ? value.xyz : AXES[channel] * value[channel];
            sample_color = 0.5 + 0.5 * centered;
            amount = channel == ALL ? length(value.xyz) : abs(value[channel]);
        } else {
            sample_color = channel == ALL ? value.xyz : vec3(value[channel]);
            amount = channel == ALL ? max(max(value.x, value.y), value.z) : value[channel];
        }
        float alpha = 1.0 - exp(-max(amount, 0.0) * density * step_length);
        color += (1.0 - coverage) * alpha * clamp(sample_color, 0.0, 1.0);
        coverage += (1.0 - coverage) * alpha;
    }
    fragColor = vec4(color, coverage);
}
