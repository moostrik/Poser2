#version 460 core

const int MAX_FIELDS = 16;

// effect: how a box shows its face's distance from good
const int DARK = 0;
const int PIXELS = 1;
const int BLUR = 2;
const int FOG = 3;

// fill: how the chosen face's box turns to color with its charge
const int FADE = 0;
const int RISE = 1;
const int IRIS = 2;
const int TRACE = 3;
const int SCAN = 4;

uniform sampler2D tex0;
uniform sampler2D tex_blur;         // a blurred tex0, for the blur effect
uniform vec2 resolution;            // of the output, in pixels

uniform vec4 fields[MAX_FIELDS];    // the boxes of the detected faces: x, y, width, height (normalized, top-left origin, Y-down)
uniform float field_far[MAX_FIELDS];    // each face's distance from good: 0 good .. 1 far
uniform int field_count;
uniform int effect;
uniform float amount;               // strength of the effect at distance 1

uniform vec4 rect;                  // the chosen face's box, as a field
uniform float rect_far;             // its distance from good
uniform float charge;               // 0..1 toward the flash
uniform int fill;
uniform vec4 face_color;            // of the traced outline
uniform float mosaic;               // pixels: block size of the whole image (the warm-up); 1 or less is none

in vec2 texCoord;
out vec4 fragColor;

const float EDGE = 0.04;            // softness of the color's moving edge, as a part of the rect's height
const float MAX_BLOCK = 64.0;       // pixels: block size of the pixels effect at amount 1
const float TRACE_PX = 10.0;        // width of the traced outline
const float SCAN_GLOW = 0.6;        // brightness of the scan line

float inside(vec4 box, vec2 p) {
    vec2 within = step(box.xy, p) * step(p, box.xy + box.zw);
    return within.x * within.y;
}

// black and white as the Hsv shader makes it at saturation 0: the brightest channel
float gray_of(vec3 color) {
    return max(color.r, max(color.g, color.b));
}

void main() {
    vec2 p = vec2(texCoord.x, 1.0 - texCoord.y);
    float in_rect = inside(rect, p);

    // how far from good the face of this pixel is: the worst of the boxes it lies in
    float far = in_rect * rect_far;
    for (int i = 0; i < min(field_count, MAX_FIELDS); i++) {
        far = max(far, inside(fields[i], p) * field_far[i]);
    }

    vec2 coord = texCoord;
    if (mosaic > 1.0) {
        coord = (floor(texCoord * resolution / mosaic) + 0.5) * mosaic / resolution;
    }
    vec4 color = texture(tex0, coord);
    float gray = gray_of(color.rgb);
    if (effect == DARK) {
        gray *= mix(1.0, 1.0 - amount, far);
    } else if (effect == PIXELS) {
        float block = mix(1.0, max(amount * MAX_BLOCK, 1.0), far);
        if (block > 1.0) {
            vec2 uv = (floor(texCoord * resolution / block) + 0.5) * block / resolution;
            gray = gray_of(texture(tex0, uv).rgb);
        }
    } else if (effect == BLUR) {
        gray = mix(gray, gray_of(texture(tex_blur, texCoord).rgb), far);
    } else if (effect == FOG) {
        gray = mix(gray, 0.5, far * amount);
    }

    // the color of the chosen face: how much of it at this pixel
    float level = charge * (1.0 + EDGE);
    float from_bottom = (rect.y + rect.w - p.y) / max(rect.w, 1e-6);
    float color_amount = charge;
    float glow = 0.0;
    if (fill == RISE || fill == SCAN) {
        // full color below a line that climbs from the bottom (charge 0) to the top (1)
        color_amount = 1.0 - smoothstep(level - EDGE, level, from_bottom);
        if (fill == SCAN) {
            float line = 1.0 - smoothstep(0.0, EDGE, abs(from_bottom - (level - EDGE * 0.5)));
            glow = SCAN_GLOW * line * step(0.001, charge) * step(charge, 0.999);
        }
    } else if (fill == IRIS) {
        // a circle from the center that reaches the corners at charge 1
        vec2 from_center = (p - (rect.xy + rect.zw * 0.5)) * resolution;
        float radius = length(from_center) / max(length(rect.zw * resolution * 0.5), 1e-6);
        color_amount = 1.0 - smoothstep(level - EDGE, level, radius);
    } else if (fill == TRACE) {
        color_amount = 0.0;       // the outline alone shows the charge
    }
    vec3 result = mix(vec3(gray), color.rgb, color_amount * in_rect) + glow * in_rect;

    if (fill == TRACE && in_rect > 0.0) {
        // the outline draws itself from the top center, clockwise, up to charge of the perimeter
        vec2 size = rect.zw * resolution;
        vec2 q = (p - rect.xy) * resolution;
        float top = q.y, right = size.x - q.x, bottom = size.y - q.y, left = q.x;
        float nearest = min(min(top, bottom), min(left, right));
        float along;
        if (nearest == top) {
            along = q.x >= size.x * 0.5 ? q.x - size.x * 0.5 : 2.0 * (size.x + size.y) - (size.x * 0.5 - q.x);
        } else if (nearest == right) {
            along = size.x * 0.5 + q.y;
        } else if (nearest == bottom) {
            along = size.x * 0.5 + size.y + (size.x - q.x);
        } else {
            along = size.x * 1.5 + size.y + (size.y - q.y);
        }
        float traced = step(nearest, TRACE_PX) * step(along / (2.0 * (size.x + size.y)), charge) * step(0.001, charge);
        result = mix(result, face_color.rgb, traced * face_color.a);
    }

    fragColor = vec4(result, color.a);
}
