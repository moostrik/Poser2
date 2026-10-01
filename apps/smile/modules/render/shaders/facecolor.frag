#version 460 core

uniform sampler2D tex0;
uniform vec4 rect;          // x, y, width, height (normalized 0..1, top-left origin, Y-down)
uniform float saturation;   // colour inside the rect: 0 black and white, 1 full colour
uniform float rise;         // 0: the colour fades in over the whole rect, 1: it rises from the rect's bottom

in vec2 texCoord;
out vec4 fragColor;

const float EDGE = 0.04;    // softness of the rising colour's top edge, as a part of the rect's height

void main() {
    vec4 color = texture(tex0, texCoord);
    // black and white as the Hsv shader makes it at saturation 0: the brightest channel
    float gray = max(color.r, max(color.g, color.b));

    vec2 p = vec2(texCoord.x, 1.0 - texCoord.y);
    vec2 inside = step(rect.xy, p) * step(p, rect.xy + rect.zw);
    // rise: full colour below a line that climbs from the bottom (saturation 0) to the top (1)
    float from_bottom = (rect.y + rect.w - p.y) / max(rect.w, 1e-6);
    float level = saturation * (1.0 + EDGE);
    float risen = 1.0 - smoothstep(level - EDGE, level, from_bottom);
    float amount = mix(saturation, risen, rise) * inside.x * inside.y;

    fragColor = vec4(mix(vec3(gray), color.rgb, amount), color.a);
}
