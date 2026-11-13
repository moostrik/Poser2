#version 460 core

// Volume Cube: a volume drawn continuously in its box, one ray per pixel from the orbit camera,
// marched front to back. Each sample is colored by `mode` (as stored; a vector by its direction alone;
// a scalar centered on grey) and its amount, capped at 1, is its opacity: an image's largest channel,
// a scalar's or a vector's size. Far samples are dimmed by depth_fade. With the slice on, the ray reads
// the selected layer where it crosses it, and slice_focus moves opacity from the rest to it: at 0 the
// slice is a sample as any other, at 1 it shows its amount as its opacity and the rest none. Out comes
// the premultiplied color and its coverage

uniform sampler3D tex0;               // The volume
uniform mat4 inverse_view_projection; // clip space back to the box's space
uniform vec3 box;                     // the box's size, its width 1
uniform float scale;                  // Multiplier on the values; at 1 after it, a sample is full
uniform int channel;                  // 0 to 3 for x, y, z, w; 4 for all
uniform int mode;                     // 0 as stored, 1 a vector by its direction, 2 centered on grey
uniform int steps;                    // samples along a ray through the whole box
uniform float opacity;                // opacity per box width of a full sample
uniform float depth_fade;             // dimming per box width behind the box's near side
uniform bool slice_on;                // the selected layer in focus, the rest dimmed
uniform float slice_z;                // the selected layer's depth, 0 front to 1 back
uniform float slice_focus;            // 0 the slice as any sample, 1 the slice at its amount and the rest at none

in vec2 texCoord;
out vec4 fragColor;

const int ALL = 4;
const int AS_STORED = 0;
const int DIRECTION = 1;
const vec3 AXES[4] = vec3[](vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(1.0));

vec3 unproject(vec2 ndc, float depth) {
    vec4 p = inverse_view_projection * vec4(ndc, depth, 1.0);
    return p.xyz / p.w;
}

// A sample's color and its amount, capped at 1
vec4 shade(vec3 uvw) {
    vec4 value = texture(tex0, uvw) * scale;
    vec3 sample_color;
    float amount;
    if (mode == DIRECTION) {
        vec3 v = channel == ALL ? value.xyz : AXES[channel] * value[channel];
        float size = length(v);
        sample_color = size > 0.0 ? 0.5 + 0.5 * v / size : vec3(0.5);
        amount = size;
    } else if (mode == AS_STORED) {
        sample_color = channel == ALL ? value.xyz : vec3(value[channel]);
        amount = channel == ALL ? max(max(value.x, value.y), value.z) : value[channel];
    } else {
        vec3 centered = channel == ALL ? value.xyz : AXES[channel] * value[channel];
        sample_color = 0.5 + 0.5 * centered;
        amount = channel == ALL ? length(value.xyz) : abs(value[channel]);
    }
    return vec4(clamp(sample_color, 0.0, 1.0), clamp(amount, 0.0, 1.0));
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

    // Where the ray crosses the selected layer, if it does inside the box
    float slice_t = -1.0;
    if (slice_on && abs(dir.z) > 1e-6) {
        float t = (box.z * (0.5 - slice_z) - origin.z) / dir.z;
        if (t >= enter && t <= leave) slice_t = t;
    }

    float step_length = length(box) / float(steps);
    vec3 color = vec3(0.0);
    float coverage = 0.0;
    bool sliced = slice_t < 0.0;
    for (float t = enter + 0.5 * step_length; t < leave && coverage < 0.99; t += step_length) {
        if (!sliced && t >= slice_t) {          // the layer itself, from a sample's opacity toward its amount
            vec3 p = origin + dir * slice_t;
            vec4 s = shade(vec3(p.x / box.x + 0.5, p.y / box.y + 0.5, slice_z));
            float fade = exp(-depth_fade * (slice_t - enter));
            float alpha = mix(1.0 - exp(-s.a * opacity * step_length), s.a, slice_focus);
            color += (1.0 - coverage) * alpha * fade * s.rgb;
            coverage += (1.0 - coverage) * alpha;
            sliced = true;
        }
        vec3 p = origin + dir * t;
        vec3 uvw = vec3(p.x / box.x + 0.5, p.y / box.y + 0.5, 0.5 - p.z / box.z);   // the front, layer 0, toward +z
        vec4 s = shade(uvw);
        float share = slice_on ? 1.0 - slice_focus : 1.0;    // the rest's opacity, what the slice takes
        float alpha = (1.0 - exp(-s.a * opacity * step_length)) * share;
        float fade = exp(-depth_fade * (t - enter));
        color += (1.0 - coverage) * alpha * fade * s.rgb;
        coverage += (1.0 - coverage) * alpha;
    }
    fragColor = vec4(color, coverage);
}
