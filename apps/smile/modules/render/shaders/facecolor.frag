#version 460 core

const int MAX_FIELDS = 16;

uniform sampler2D tex0;
uniform vec4 rect;          // x, y, width, height (normalized 0..1, top-left origin, Y-down)
uniform float saturation;   // colour inside the rect: 0 black and white, 1 full colour
uniform float rise;         // 0: the colour fades in over the whole rect, 1: it rises from the rect's bottom
uniform vec4 fields[MAX_FIELDS];    // the boxes of the detected faces, as rect
uniform int field_count;
uniform float dark;         // brightness of the fields and of the rect where it has no colour (1: not darkened)

in vec2 texCoord;
out vec4 fragColor;

const float EDGE = 0.04;    // softness of the rising colour's top edge, as a part of the rect's height

float inside(vec4 box, vec2 p) {
    vec2 within = step(box.xy, p) * step(p, box.xy + box.zw);
    return within.x * within.y;
}

void main() {
    vec4 color = texture(tex0, texCoord);
    // black and white as the Hsv shader makes it at saturation 0: the brightest channel
    float gray = max(color.r, max(color.g, color.b));

    vec2 p = vec2(texCoord.x, 1.0 - texCoord.y);
    float in_rect = inside(rect, p);
    float in_field = in_rect;
    for (int i = 0; i < min(field_count, MAX_FIELDS); i++) {
        in_field = max(in_field, inside(fields[i], p));
    }
    vec3 base = vec3(gray) * mix(1.0, dark, in_field);

    // rise: full colour below a line that climbs from the bottom (saturation 0) to the top (1)
    float from_bottom = (rect.y + rect.w - p.y) / max(rect.w, 1e-6);
    float level = saturation * (1.0 + EDGE);
    float risen = 1.0 - smoothstep(level - EDGE, level, from_bottom);
    float amount = mix(saturation, risen, rise) * in_rect;

    fragColor = vec4(mix(base, color.rgb, amount), color.a);
}
